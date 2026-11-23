"""Bank reconciliation - Streamlit app (V1, no storage).

Run locally:  streamlit run app.py
Everything lives in this browser session's memory and is gone when the tab closes.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from recon import MatchConfig, run_matching
from recon.autodetect import DATE_FORMATS, decode, guess_date_format, guess_mapping, read_head
from recon.normalize import normalize, parse_amount
from recon.report import (BUCKET_LABELS, LAYER_LABELS, build_reconciliation, exceptions_csv,
                          journal_csv, journal_rows, matches_csv, money, render_html)
from recon.explain import REPO, diagram_html, funnel, stages
from recon.review import accept_suggestion, reopen, resolve

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data"
NONE = "(none)"
SIDES = {"bank": "Bank statement", "ledger": "Books (general ledger)"}
LABEL_TO_BUCKET = {v: k for k, v in BUCKET_LABELS.items() if k}
BLANK = "—"
REASON_CHOICES = [BLANK] + list(LABEL_TO_BUCKET)
ACTIONS = [BLANK, "Accept best suggestion", "Mark resolved", "Reopen"]
PRIVACY = ("Nothing is saved. Your files and decisions stay in memory for this browser tab only "
           "and are gone when you close or refresh it, so download your report before leaving. "
           "On the hosted demo, files pass through Streamlit's servers while you work.")

st.set_page_config(page_title="Bank reconciliation", layout="wide")
ss = st.session_state
ss.setdefault("files", {})
ss.setdefault("maps", {})
ss.setdefault("sample_meta", None)
ss.setdefault("result", None)
ss.setdefault("ev", 0)
ss.setdefault("flash", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_cents(x: float | None) -> int | None:
    return None if x is None else round(x * 100)


def load_sample() -> None:
    ss.files = {
        "bank": {"name": "sample_bank.csv", "text": (SAMPLE / "bank.csv").read_text()},
        "ledger": {"name": "sample_ledger.csv", "text": (SAMPLE / "ledger.csv").read_text()},
    }
    ss.sample_meta = json.loads((SAMPLE / "meta.json").read_text())
    ss.result = None


def start_over() -> None:
    for k in ("files", "maps", "sample_meta", "result"):
        ss[k] = {} if k in ("files", "maps") else None
    ss.ev += 1


def idx(options: list, value) -> int:
    return options.index(value) if value in options else 0


def mapping_form(source: str, text: str) -> dict:
    """Preview plus column pickers, pre-filled with a best guess."""
    key = f"{source}_{hashlib.md5(text.encode()).hexdigest()[:8]}_"
    header, rows = read_head(text, 50)
    g = guess_mapping(text, source)
    st.dataframe(pd.DataFrame(rows[:5]), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date column", header, index=idx(header, g["date"]["column"]), key=key + "date")
    fmt_guess = guess_date_format([r.get(date_col, "") for r in rows]) or g["date"]["format"]
    fmt = c2.selectbox("Date format", DATE_FORMATS, index=idx(DATE_FORMATS, fmt_guess), key=key + "fmt",
                       format_func=lambda f: f"{date(2026, 11, 3).strftime(f)}   ({f})")

    modes = ["One amount column", "Separate debit and credit columns"]
    mode = st.radio("How are amounts shown?", modes, horizontal=True, key=key + "mode",
                    index=1 if "debit" in g["amount"] else 0)
    if mode == modes[0]:
        col = st.selectbox("Amount column", header, index=idx(header, g["amount"].get("column")),
                           key=key + "amt")
        flip = st.checkbox("Positive numbers mean money going out", key=key + "flip",
                           help="Most exports show money in as positive. Tick this if yours is the other way round.")
        amount = {"column": col, "sign": -1 if flip else 1}
    else:
        c1, c2 = st.columns(2)
        deb = c1.selectbox("Debit column", header, index=idx(header, g["amount"].get("debit")), key=key + "deb")
        cre = c2.selectbox("Credit column", header, index=idx(header, g["amount"].get("credit")), key=key + "cre")
        inflow = st.checkbox("Debit means money coming in", value=source == "ledger", key=key + "inflow",
                             help="True for the cash account in your books. On a bank statement, debit means money out.")
        amount = {"debit": deb, "credit": cre, "debit_is_inflow": inflow}

    opts = [NONE] + header
    c1, c2, c3 = st.columns(3)
    desc = c1.selectbox("Description column", header, index=idx(header, g["description"]), key=key + "desc")
    ref = c2.selectbox("Reference ID column", opts, index=idx(opts, g["reference"] or NONE), key=key + "ref",
                       help="Transaction or check numbers that also appear on the other side. "
                            "The more lines have one, the more matches are certain.")
    mapping = {"source": source, "date": {"column": date_col, "format": fmt}, "amount": amount,
               "description": desc, "reference": None if ref == NONE else ref}
    if source == "bank":
        bal = c3.selectbox("Running balance column", opts, index=idx(opts, g["balance"] or NONE),
                           key=key + "bal", help="Optional. Used to fill in the statement's ending balance.")
        mapping["balance"] = None if bal == NONE else bal
    else:
        ext = c3.selectbox("Entry ID column", opts, index=idx(opts, g["external_id"] or NONE), key=key + "ext")
        mapping["external_id"] = None if ext == NONE else ext
    return mapping


def ending_bank_balance(text: str, mapping: dict) -> float | None:
    """Last running balance on the statement, if the file has one."""
    col = mapping.get("balance")
    if not col:
        return None
    res = normalize(text, mapping)
    if not res.transactions:
        return None
    last = max(res.transactions, key=lambda t: (t.txn_date, t.source_row))
    _, rows = read_head(text, n=10**7)
    try:
        cents = parse_amount(rows[last.source_row - 1][col])
    except (ValueError, IndexError, KeyError):
        return None
    return None if cents is None else cents / 100


def describe(ids: list[str], by_id: dict) -> str:
    parts = []
    for i in ids:
        t = by_id[i]
        side = "Bank" if t.source == "bank" else "Books"
        parts.append(f"{side} {t.txn_date:%b %d} {t.amount:,.2f} {t.description[:34]}")
    return " + ".join(parts)


def suggestion_text(e, by_id: dict) -> str:
    if not e.suggestions:
        return ""
    s = e.suggestions[0]
    others = [i for i in s.bank_ids + s.ledger_ids if i != e.transaction_id]
    tag = " (digits swapped?)" if s.transposition else ""
    return f"{describe(others, by_id)}  [{s.confidence:.0f}]{tag}"


def how_it_works(cfg: MatchConfig, result=None) -> None:
    """The pipeline, explained from the settings and results actually used."""
    st.write(f"This tool is open source. Every step below runs the code in "
             f"[the GitHub repository]({REPO}), with the settings shown here"
             + (" for this run." if result else ". Numbers appear after you reconcile."))
    st.html(diagram_html(result, cfg))

    if result:
        st.subheader("What happened in this run")
        names = {**LAYER_LABELS, "manual": "Confirmed by you"}
        rows = [{"Step": names.get(r["stage"], r["stage"]), "Lines coming in": r["in"],
                 "Matched here": r["matched"], "Left for the next step": r["left"]}
                for r in funnel(result)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Counts are lines from both files. A match of one bank line to one book line counts as 2.")

    st.subheader("Each step, in plain language")
    for stg in stages(result.config if result else cfg):
        with st.expander(stg.title):
            st.markdown(f"**What it does.** {stg.does}")
            st.markdown(f"**The rule.** {stg.rule}")
            st.markdown(f"**What it never does.** {stg.never}")
            st.markdown(f"[Read the code]({stg.source})")

    with st.expander("All settings" + (" used in this run" if result else "")):
        used = (result.config if result else cfg).to_dict()
        st.dataframe(pd.DataFrame([{"Setting": k, "Value": str(v)} for k, v in used.items()]),
                     hide_index=True, width="stretch")
        st.caption("Change the main ones under Matching settings in the sidebar.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Bank reconciliation")
    st.caption("Match your bank statement to your books, see what doesn't line up, "
               "and download a reconciliation report.")
    if st.button("Try it with sample data", width="stretch"):
        load_sample()
        st.rerun()
    if st.button("Start over", width="stretch"):
        start_over()
        st.rerun()
    with st.expander("Matching settings"):
        exact_w = st.slider("Days the bank may post after the books (exact match)", 0, 15, 5)
        fuzzy_w = st.slider("Days to search for close matches", exact_w, 30, max(10, exact_w))
        auto = st.slider("Confidence needed to match automatically", 70, 100, 90)
        group = st.slider("Most lines in one grouped match", 2, 6, 5)
        st.caption("Changes apply the next time you reconcile.")
    st.info(PRIVACY, icon=":material/lock:")

cfg = MatchConfig(exact_window_days=exact_w, fuzzy_window_days=fuzzy_w, auto_accept=float(auto),
                  max_group_size=group)

for kind, msg in ss.flash:
    getattr(st, kind)(msg)
ss.flash = []

# ---------------------------------------------------------------------------
# Step 1: upload and map columns
# ---------------------------------------------------------------------------

if ss.result is None:
    st.title("Reconcile your bank account")
    st.write("Upload a bank statement and an export of the same account from your books, both as CSV. "
             "Check that the columns were recognized, then press **Reconcile**.")

    cols = st.columns(2)
    for col, source in zip(cols, ("bank", "ledger")):
        with col:
            st.subheader(SIDES[source])
            up = st.file_uploader(f"{SIDES[source]} (CSV)", type=["csv"], key=f"up_{source}_{ss.ev}")
            if up is not None:
                ss.files[source] = {"name": up.name, "text": decode(up.getvalue())}
            if source in ss.files:
                f = ss.files[source]
                st.caption(f"Using **{f['name']}**")
                with st.expander("Check columns", expanded=ss.sample_meta is None):
                    ss.maps[source] = mapping_form(source, f["text"])

    if len(ss.files) == 2:
        bank_res = normalize(ss.files["bank"]["text"], ss.maps["bank"])
        led_res = normalize(ss.files["ledger"]["text"], ss.maps["ledger"])
        all_dates = [t.txn_date for t in bank_res.transactions + led_res.transactions]
        default_end = (date.fromisoformat(ss.sample_meta["period"]["end"]) if ss.sample_meta
                       else max(all_dates) if all_dates else date.today())
        c1, c2 = st.columns([1, 2])
        period_end = c1.date_input("Period end date", value=default_end,
                                   help="Usually the statement's closing date.")
        problems = [(SIDES[r.transactions[0].source if r.transactions else src], i)
                    for src, r in (("bank", bank_res), ("ledger", led_res)) for i in r.issues]
        if problems:
            st.error("Some rows couldn't be read with these column settings:")
            for side, i in problems[:10]:
                st.write(f"- {side}, {'row ' + str(i.row) if i.row else 'file'}: {i.message}")
            if len(problems) > 10:
                st.write(f"...and {len(problems) - 10} more.")
        if st.button("Reconcile", type="primary", disabled=bool(problems)):
            with st.spinner("Matching..."):
                ss.result = run_matching(bank_res.transactions, led_res.transactions, cfg, period_end)
            ss.period_end = period_end
            ss.bank_bal = ending_bank_balance(ss.files["bank"]["text"], ss.maps["bank"])
            ss.book_bal = (float(ss.sample_meta["balances"]["ledger_closing"]) if ss.sample_meta else None)
            ss.ev += 1
            st.rerun()
    else:
        st.info("Upload both files, or use **Try it with sample data** in the sidebar.")
    st.divider()
    with st.expander("How the matching works (open source)"):
        how_it_works(cfg)
    st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

r = ss.result
by_id = {t.id: t for t in r.transactions}
s = r.summary
open_exc = [e for e in r.exceptions if not e.resolved]

head, btn = st.columns([4, 1])
head.title("Reconciliation")
head.caption(f"Period ending {ss.period_end:%B %d, %Y}  ·  {ss.files['bank']['name']}  vs  "
             f"{ss.files['ledger']['name']}")
if btn.button("Change files or settings"):
    ss.result = None
    st.rerun()

tab_sum, tab_exc, tab_match, tab_rep, tab_how = st.tabs(
    ["Summary", f"Needs attention ({len(open_exc)})", f"Matched ({len(r.matches)})", "Report and download",
     "How it works"])

# --- Summary ------------------------------------------------------------------
with tab_sum:
    total = s["transactions"]["bank"] + s["transactions"]["ledger"]
    matched = s["matched"]["bank"] + s["matched"]["ledger"]
    open_value = sum(abs(by_id[e.transaction_id].amount_cents) for e in open_exc)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Lines matched ({matched} of {total})", f"{s['match_rate']:.0%}")
    m2.metric("Need attention", len(open_exc))
    m3.metric("Resolved by you", s["exceptions"]["resolved"] + s["matches_by_layer"].get("manual", 0))
    m4.metric("Open items value", money(open_value))

    left, right = st.columns(2)
    with left:
        st.subheader("How matches were made")
        order = ["reference_id", "exact", "fuzzy", "combination", "manual"]
        shares = s["matched_bank_share_by_layer"]
        method_rows = [{"Method": LAYER_LABELS[k], "Matches": s["matches_by_layer"].get(k, 0),
                        "Share of matched bank lines": shares[k] * 100}
                       for k in order if k in shares]
        if method_rows:
            st.dataframe(pd.DataFrame(method_rows), hide_index=True, width="stretch",
                         column_config={"Share of matched bank lines": st.column_config.ProgressColumn(
                             min_value=0, max_value=100, format="%.0f%%")})
        cov = s["reference_coverage"]
        st.caption(f"Reference IDs found on {cov['bank']:.0%} of bank lines and {cov['ledger']:.0%} of "
                   f"book lines. More shared IDs mean more matches that are certain.")
    with right:
        st.subheader("Items by reason")
        reasons = {}
        for e in r.exceptions:
            label = BUCKET_LABELS.get(e.bucket, e.bucket)
            row = reasons.setdefault(label, {"Open": 0, "Resolved": 0})
            row["Resolved" if e.resolved else "Open"] += 1
        if reasons:
            st.dataframe(pd.DataFrame(reasons).T, width="stretch")
        else:
            st.success("Nothing left to review.")

# --- Needs attention ---------------------------------------------------------
with tab_exc:
    st.write("Each row is a line that didn't match automatically. Pick an **Action**, "
             "set a **Reason** when resolving, then press **Apply**.")
    view = st.radio("Show", ["Open", "Resolved", "All"], horizontal=True, label_visibility="collapsed")
    shown = [e for e in r.exceptions if view == "All" or (view == "Open") != e.resolved]
    if not shown:
        st.success("Nothing here.")
    else:
        rows = []
        for e in shown:
            t = by_id[e.transaction_id]
            rows.append({
                "id": e.transaction_id,
                "Side": "Bank" if t.source == "bank" else "Books",
                "Date": t.txn_date,
                "Amount": t.amount,
                "Description": t.description,
                "Action": BLANK,
                "Reason": BUCKET_LABELS.get(e.bucket, BLANK) if e.bucket else BLANK,
                "Note": e.note,
                "Best match found": suggestion_text(e, by_id),
                "Why": e.reason,
            })
        df = pd.DataFrame(rows).set_index("id")
        edited = st.data_editor(
            df, key=f"exc_{ss.ev}_{view}", hide_index=True, width="stretch",
            disabled=["Side", "Date", "Amount", "Description", "Why", "Best match found"],
            column_config={
                "Side": st.column_config.TextColumn(width=60),
                "Date": st.column_config.DateColumn(format="MMM DD", width=70),
                "Amount": st.column_config.NumberColumn(format="%.2f", width=90),
                "Description": st.column_config.TextColumn(width=210),
                "Action": st.column_config.SelectboxColumn(
                    options=ACTIONS, width=190,
                    help="Accept best suggestion: confirm the match in 'Best match found'. "
                         "Mark resolved: keep it as an open item with the chosen reason."),
                "Reason": st.column_config.SelectboxColumn(
                    options=REASON_CHOICES, width=170,
                    help="Pre-filled with the system's suggestion when it has one."),
                "Note": st.column_config.TextColumn(width=150),
                "Best match found": st.column_config.TextColumn(width=330),
                "Why": st.column_config.TextColumn(width=330),
            })
        if st.button("Apply", type="primary"):
            done, problems = 0, []
            for tid, row in edited.iterrows():
                action, note = row["Action"], (row["Note"] or "")
                try:
                    if action == "Accept best suggestion":
                        accept_suggestion(r, tid, 0, note)
                        done += 1
                    elif action == "Mark resolved":
                        bucket = LABEL_TO_BUCKET.get(row["Reason"])
                        if not bucket:
                            problems.append(f"{row['Description']}: choose a reason first.")
                            continue
                        resolve(r, tid, bucket, note)
                        done += 1
                    elif action == "Reopen":
                        reopen(r, tid)
                        done += 1
                except (ValueError, KeyError) as err:
                    problems.append(f"{row['Description']}: {err}")
            if done:
                ss.flash.append(("success", f"Applied {done} decision(s)."))
            for p in problems:
                ss.flash.append(("warning", p))
            ss.ev += 1
            st.rerun()

    with st.expander("See every suggestion for one item"):
        with_s = [e for e in r.exceptions if e.suggestions]
        if not with_s:
            st.caption("No open item has suggestions.")
        else:
            pick = st.selectbox("Item", with_s, format_func=lambda e: describe([e.transaction_id], by_id))
            for n, sg in enumerate(pick.suggestions):
                others = [i for i in sg.bank_ids + sg.ledger_ids if i != pick.transaction_id]
                c1, c2 = st.columns([5, 1])
                c1.write(f"**{describe(others, by_id)}**  \n{sg.reason}")
                if c2.button("Accept", key=f"acc_{pick.transaction_id}_{n}_{ss.ev}",
                             disabled=sg.transposition):
                    try:
                        accept_suggestion(r, pick.transaction_id, n)
                        ss.flash.append(("success", "Match confirmed."))
                    except (ValueError, KeyError) as err:
                        ss.flash.append(("warning", str(err)))
                    ss.ev += 1
                    st.rerun()

# --- Matched -----------------------------------------------------------------
with tab_match:
    layers = [LAYER_LABELS.get(k, k) for k in s["matches_by_layer"]]
    chosen = st.multiselect("How matched", layers, default=layers)
    rows = []
    for m in r.matches:
        how = LAYER_LABELS.get(m.layer, m.layer)
        if how not in chosen:
            continue
        bank_ids = [i for i in m.transaction_ids if i.startswith("bank:")]
        led_ids = [i for i in m.transaction_ids if i.startswith("ledger:")]
        rows.append({
            "How": how,
            "Confidence": m.confidence,
            "Amount": sum(by_id[i].amount for i in bank_ids),
            "Bank": " + ".join(f"{by_id[i].txn_date:%b %d} {by_id[i].description}" for i in bank_ids),
            "Books": " + ".join(f"{by_id[i].txn_date:%b %d} {by_id[i].description}" for i in led_ids),
            "Why": m.reason,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Amount": st.column_config.NumberColumn(format="%.2f"),
                                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                              format="%.0f")})

# --- Report and download -------------------------------------------------------
with tab_rep:
    c1, c2 = st.columns(2)
    bank_bal = c1.number_input("Bank statement ending balance", value=ss.get("bank_bal"), step=100.0,
                               format="%.2f", placeholder="From your bank statement")
    book_bal = c2.number_input("Book balance at period end", value=ss.get("book_bal"), step=100.0,
                               format="%.2f", placeholder="From your accounting software")
    ss.bank_bal, ss.book_bal = bank_bal, book_bal

    if bank_bal is None or book_bal is None:
        st.info("Enter both ending balances to build the reconciliation report.")
    else:
        rec = build_reconciliation(r, to_cents(bank_bal), to_cents(book_bal))
        msg = {
            "reconciled": ("success", "Reconciled. The adjusted balances agree and every item has a decision."),
            "balanced_open_items": ("info", f"The adjusted balances agree, but {len(rec.unresolved)} item(s) "
                                            f"still need a decision. They currently offset each other."),
            "unresolved_items": ("warning", f"Not reconciled yet. The {money(rec.difference)} difference comes "
                                            f"from {len(rec.unresolved)} item(s) waiting for a decision."),
            "unexplained": ("error", f"Not reconciled. {money(rec.difference)} isn't explained by open items. "
                                     f"Check the balances you entered, or look for missing lines."),
        }[rec.status]
        getattr(st, msg[0])(msg[1])

        m1, m2, m3 = st.columns(3)
        m1.metric("Adjusted bank balance", money(rec.adjusted_bank))
        m2.metric("Adjusted book balance", money(rec.adjusted_book))
        m3.metric("Difference", money(rec.difference))

        def item_df(items):
            return pd.DataFrame([{"Date": i.txn_date, "Item": i.label, "Amount": i.amount_cents / 100,
                                  "Description": i.description} for i in items])

        item_cols = {"Date": st.column_config.DateColumn(format="MMM DD", width=80),
                     "Item": st.column_config.TextColumn(width=280),
                     "Amount": st.column_config.NumberColumn(format="%.2f", width=110)}
        st.subheader("Bank side")
        st.caption("Starts from the bank balance, adds deposits in transit and subtracts payments not yet cleared.")
        if rec.bank_items:
            st.dataframe(item_df(rec.bank_items), hide_index=True, width="stretch", column_config=item_cols)
        else:
            st.caption("No adjustments.")
        st.subheader("Book side")
        st.caption("Starts from the book balance, adds what the bank shows but the books don't, and corrections.")
        if rec.book_items:
            st.dataframe(item_df(rec.book_items), hide_index=True, width="stretch", column_config=item_cols)
        else:
            st.caption("No adjustments.")
        if rec.unresolved:
            st.subheader("Still open")
            st.caption("Items without a decision. Resolve them in 'Needs attention'.")
            st.dataframe(item_df(rec.unresolved), hide_index=True, width="stretch", column_config=item_cols)

        st.subheader("Suggested journal entries")
        jr = journal_rows(rec)
        if jr:
            st.dataframe(pd.DataFrame(jr), hide_index=True, width="stretch")
            st.caption("Review before posting. Accounts in quotes like 'Account on entry JE-...' mean "
                       "use the same account as that original entry.")
        else:
            st.caption("No entries needed.")

        st.subheader("Download")
        stamp = ss.period_end.isoformat()
        report_html = render_html(r, rec, ss.period_end)
        files = {
            f"reconciliation_{stamp}.html": report_html,
            f"journal_entries_{stamp}.csv": journal_csv(rec),
            f"matches_{stamp}.csv": matches_csv(r),
            f"open_items_{stamp}.csv": exceptions_csv(r),
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, content in files.items():
                z.writestr(name, content)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Everything (.zip)", buf.getvalue(), f"reconciliation_{stamp}.zip",
                           "application/zip", type="primary", width="stretch")
        d2.download_button("Report (.html)", report_html, f"reconciliation_{stamp}.html", "text/html",
                           width="stretch")
        d3.download_button("Journal entries (.csv)", files[f"journal_entries_{stamp}.csv"],
                           f"journal_entries_{stamp}.csv", "text/csv", width="stretch")
        d4.download_button("Matches (.csv)", files[f"matches_{stamp}.csv"], f"matches_{stamp}.csv",
                           "text/csv", width="stretch")
        st.caption("The HTML report opens in any browser and prints cleanly to PDF.")

# --- How it works ----------------------------------------------------------------
with tab_how:
    how_it_works(cfg, r)
