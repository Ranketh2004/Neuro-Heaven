import html
import streamlit as st
import streamlit.components.v1 as components
from textwrap import dedent

from utils.api_client import post, APIError


def _init_state():
    defaults = {
        "asm_show_required": False,

        "age": None,
        "age_of_onset": None,
        "pretreatment_seizure_count": None,
        "prior_asm_exposure_count": None,

        "sex": None,
        "seizure_type": None,
        "current_asm": None,
        "mri_lesion_type": None,
        "eeg_status_detail": None,

        "psychiatric_disorder": None,
        "intellectual_disability": None,
        "cerebrovascular_disease": None,
        "head_trauma": None,
        "cns_infection": None,
        "substance_alcohol_abuse": None,
        "family_history": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _is_placeholder(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s == "select" or s.startswith("select ")
    return False


def _clean_cat_keep_placeholder(v: object):
    return v


def _clean_num(v: object):
    return None if v is None else v


def _safe_index(options: list[str], value: object) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _is_missing_required(v: object) -> bool:
    return _is_placeholder(v)


def _build_next_steps(
    out: dict,
    ranking_out: dict | None,
    patient: dict | None,
) -> list:
    """Generate 5–6 clinically personalised recommended next steps."""
    p             = patient or {}
    pred_label    = int(out.get("pred_label", 0))
    prob_pct      = round(float(out.get("prob_final", 0.0)) * 100)
    ml            = out.get("ml_details", {})
    rule_pen      = float(ml.get("rule_penalty", 0.0))
    flags         = ml.get("reliability_flags", [])
    applicability = int(out.get("applicability_indicator", 0))

    def _s(v):  return str(v or "").strip().lower()
    def _yn(v): return _s(v) in {"yes", "y"}

    age          = p.get("age")
    sex          = _s(p.get("sex"))
    current_asm  = _s(p.get("current_asm"))
    mri          = _s(p.get("mri_lesion_type"))
    eeg          = _s(p.get("eeg_status_detail"))
    prior_asms   = p.get("prior_asm_exposure_count")
    psych        = _yn(p.get("psychiatric_disorder"))
    int_dis      = _yn(p.get("intellectual_disability"))
    cerebro      = _yn(p.get("cerebrovascular_disease"))
    head_trauma  = _yn(p.get("head_trauma"))
    cns_inf      = _yn(p.get("cns_infection"))
    alcohol      = _yn(p.get("substance_alcohol_abuse"))

    try:    age_f    = float(age) if age is not None else 99.0
    except Exception: age_f = 99.0
    try:    n_prior  = int(float(prior_asms)) if prior_asms is not None else 0
    except Exception: n_prior = 0

    is_wocbp    = (sex == "female" and age_f < 50)
    asm_disp    = current_asm.capitalize() if current_asm else "Current ASM"
    _no_inv     = {"", "select", "select_an_option", "select an option", "none"}
    mri_clean   = mri.replace("_", " ").title() if mri not in _no_inv else ""
    has_lesion  = bool(mri_clean)

    steps = []

    # ── 1. ASM dose adequacy & appropriateness ───────────────────────────
    if current_asm and rule_pen > 0.08:
        steps.append(
            f"{asm_disp} carries a clinically significant suitability penalty for this patient profile. "
            "Review whether this ASM is correctly indicated for the documented seizure type and discuss "
            "substitution with the top-ranked alternative before attributing failure to the patient."
        )
    elif current_asm and rule_pen > 0.005:
        steps.append(
            f"Confirm {asm_disp} is at target therapeutic dose and that adherence is documented "
            "(drug-level monitoring, pill count, or pharmacy refill record). Rule-based adjustments "
            "have reduced the predicted probability — address modifiable risk factors proactively."
        )
    else:
        steps.append(
            f"Confirm {asm_disp} dosing is adequate and that adherence is verified. "
            "Sub-therapeutic levels and intermittent adherence are the most common reversible causes "
            "of treatment failure and must be excluded before any management escalation."
        )

    # ── 2. EEG / MRI diagnostic completeness ────────────────────────────
    missing_inv = (
        (["EEG"] if eeg in _no_inv else []) +
        (["MRI brain"] if mri in _no_inv else [])
    )
    if missing_inv:
        steps.append(
            f"Obtain {' and '.join(missing_inv)} if not already available. "
            "Both investigations are essential to confirm the syndromic diagnosis, classify seizure type "
            "accurately, and identify any structural or epileptiform correlate that would alter treatment strategy."
        )
    elif has_lesion:
        steps.append(
            f"Correlate MRI finding ({mri_clean}) with EEG ({eeg or 'reported'}) results and clinical "
            "semiology. Structural lesions require targeted management — assess surgical remediability and "
            "refer for presurgical evaluation if pharmacological control is not achieved."
        )
    else:
        steps.append(
            f"Correlate EEG ({eeg or 'reported'}) findings with seizure semiology and imaging. "
            "Any discordance between EEG localisation and clinical features should prompt syndromic "
            "re-classification before proceeding with treatment changes."
        )

    # ── 3. Response reassessment window ─────────────────────────────────
    if pred_label == 1:
        steps.append(
            f"Reassess treatment response at 8–12 weeks. "
            f"The {prob_pct}% estimated seizure-freedom probability reflects a 12-month outcome; "
            "early seizure reduction within 4–12 weeks is an independent predictor of sustained remission. "
            "Document seizure frequency, severity, and adverse effects at each visit."
        )
    else:
        steps.append(
            f"Schedule urgent clinical review at 4–8 weeks. "
            f"The {prob_pct}% probability indicates elevated risk of 12-month treatment failure — "
            "early monitoring allows timely detection of non-response and reduces delay to appropriate "
            "escalation or specialist referral."
        )

    # ── 4. Drug resistance threshold & specialist referral ──────────────
    if n_prior >= 2:
        steps.append(
            f"Drug-resistant epilepsy criteria are met ({n_prior} prior ASMs failed, ILAE definition). "
            "Refer urgently to a tertiary epilepsy centre for comprehensive presurgical evaluation including "
            "video-EEG telemetry, neuropsychological assessment, and surgical candidacy review. "
            "Do not add further ASMs empirically without specialist input."
        )
    elif pred_label == 0 and n_prior == 1:
        steps.append(
            "If current treatment fails, drug-resistant epilepsy criteria will be met. "
            "Initiate referral planning for a tertiary epilepsy centre now — early referral reduces time "
            "to surgical evaluation and is associated with better long-term outcomes than delayed referral."
        )
    elif pred_label == 0:
        steps.append(
            "If seizures persist despite adequate dosing and confirmed adherence, escalate promptly. "
            "Avoid prolonged empirical polypharmacy; early specialist epilepsy referral is associated "
            "with better outcomes and reduces unnecessary medication exposure."
        )
    else:
        steps.append(
            "If seizure freedom is not achieved within 3–6 months on current treatment, reassess ASM "
            "suitability, dose, and adherence systematically before adding a second agent. "
            "Consult epilepsy specialist services rather than empirically escalating to polypharmacy."
        )

    # ── 5. Patient-specific safety / monitoring ──────────────────────────
    if is_wocbp and current_asm == "valproate":
        steps.append(
            "Valproate is in use in a female of childbearing potential — VALPROATE PREVENT programme "
            "compliance must be documented at every visit. Urgently discuss switching to Lamotrigine or "
            "Levetiracetam given the high teratogenic risk (neural tube defects, neurodevelopmental impairment), "
            "unless no suitable alternative achieves seizure control."
        )
    elif is_wocbp:
        steps.append(
            f"Document teratogenic risk counselling for {asm_disp} at every visit. "
            "If pregnancy is planned or possible, arrange a pre-conception neurology-obstetrics consultation "
            "to review ASM safety, prescribe high-dose folic acid (5 mg/day), and establish a monitoring plan."
        )
    elif psych:
        steps.append(
            "Monitor psychiatric symptoms at every epilepsy review — comorbid mood or anxiety disorders "
            "significantly worsen seizure outcomes and reduce adherence. Consider integrated psychiatric care; "
            "Levetiracetam may exacerbate behavioural symptoms while Lamotrigine may provide mood-stabilising benefit."
        )
    elif cerebro:
        steps.append(
            "In cerebrovascular disease, prioritise ASMs with minimal cardiac and drug-interaction risk. "
            "Avoid enzyme-inducing agents (carbamazepine, phenytoin, phenobarbital) that alter anticoagulant "
            "and antihypertensive drug levels. Optimise cardiovascular risk factor management concurrently."
        )
    elif alcohol:
        steps.append(
            "Alcohol use disorder impairs ASM adherence and lowers the seizure threshold. "
            "Engage addiction medicine support alongside epilepsy management. Avoid phenobarbital "
            "(combined CNS/respiratory depression risk); perform drug-level monitoring where fluctuations "
            "due to alcohol intake are anticipated."
        )
    elif int_dis:
        steps.append(
            "In intellectual disability, select ASMs with the lowest sedative and cognitive burden. "
            "Use carer-completed seizure diaries and validated quality-of-life measures at each review. "
            "Reassess the indication for each agent at least annually and avoid high-dose polypharmacy."
        )
    elif has_lesion:
        steps.append(
            f"With a structural lesion ({mri_clean}), conduct multidisciplinary team review to assess "
            "surgical candidacy. Structural epilepsies have better outcomes with lesionectomy when "
            "pharmacological remission is not achieved; early surgical referral is recommended after two ASM failures."
        )
    elif cns_inf or head_trauma:
        cause = "CNS infection" if cns_inf else "head trauma"
        steps.append(
            f"In post-{cause} epilepsy, confirm the underlying aetiology is fully addressed. "
            "Arrange repeat MRI at 12–24 months if initial imaging was normal or equivocal. "
            "Consider neuropsychology referral to assess and manage cognitive sequelae."
        )
    else:
        steps.append(
            "Record adverse effects and quality-of-life impact (QOLIE-31 or similar) at every review — "
            "treatment success is not defined by seizure count alone but by the balance of seizure control "
            "and treatment tolerability. Patient-reported outcomes should guide escalation decisions."
        )

    # ── 6. Data completeness / reliability ──────────────────────────────
    if applicability <= 1:
        steps.append(
            "Applicability of this prediction is limited — key clinical inputs (pretreatment seizure count, "
            "EEG findings, MRI, prior ASM history) are incomplete or absent. "
            "Complete the clinical dataset and re-run the prediction before using these results to guide "
            "management decisions."
        )
    elif flags:
        steps.append(
            "Data quality flags have been raised (see Flags panel). Resolve flagged inconsistencies "
            "before relying on this estimate — inaccurate clinical inputs directly reduce the validity "
            "of the prediction and the reliability of the ASM ranking."
        )

    return steps


def _render_combined_card(out: dict, ranking_out: dict | None, patient: dict | None = None) -> str:
    """Single unified HTML card: prediction result + ASM ranking."""

    # ── core values ──────────────────────────────────────────────────────
    prob        = float(out.get("prob_final", 0.0))
    prob_pct    = round(prob * 100)
    pred_label  = int(out.get("pred_label", 0))
    ml          = out.get("ml_details", {})
    model_name  = html.escape(str(ml.get("model_name", "ML Model")))
    prob_model  = float(ml.get("prob_model", prob))
    rule_pen    = float(ml.get("rule_penalty", 0.0))
    threshold   = float(ml.get("threshold", 0.5))
    threshold_pct = round(threshold * 100)
    applicability = int(out.get("applicability_indicator", 0))

    is_fav      = pred_label == 1
    banner_bar  = "#16a34a" if is_fav else "#dc2626"
    badge_text  = "Likely seizure-free" if is_fav else "Likely not seizure-free"
    badge_bg    = "rgba(220,252,231,1)" if is_fav else "rgba(254,226,226,1)"
    badge_fg    = "#166534" if is_fav else "#991b1b"

    # probability ring (SVG)
    ring_color  = "#22c55e" if prob_pct >= 60 else "#f59e0b" if prob_pct >= 36 else "#ef4444"
    circ        = 251.33
    dash        = round(circ * prob_pct / 100, 2)
    ring_svg    = f'<svg viewBox="0 0 100 100" width="88" height="88" style="display:block;"><circle cx="50" cy="50" r="40" fill="none" stroke="#e2e8f0" stroke-width="10"/><circle cx="50" cy="50" r="40" fill="none" stroke="{ring_color}" stroke-width="10" stroke-dasharray="{dash} {circ}" stroke-linecap="round" transform="rotate(-90 50 50)"/><text x="50" y="50" text-anchor="middle" dominant-baseline="central" font-size="22" font-weight="900" fill="#0f172a" font-family="system-ui,sans-serif">{prob_pct}%</text></svg>'

    # applicability dots
    app_lbl_map = {0: "Very low", 1: "Low", 2: "Moderate", 3: "Good", 4: "High"}
    app_lbl     = app_lbl_map.get(min(applicability, 4), "Very low")
    app_dots    = "".join(
        f'<span class="ap-dot ap-dot-{"on" if j < applicability else "off"}"></span>'
        for j in range(5)
    )

    # rule adjustment pill
    if abs(rule_pen) > 0.005:
        sign = "+" if rule_pen < 0 else "−"
        radj = f'<span class="pill pill-{"pos" if rule_pen < 0 else "neg"}">{sign}{abs(rule_pen)*100:.0f}% rule adj.</span>'
    else:
        radj = '<span class="pill pill-neutral">No rule adjustment</span>'

    # impression paragraph (built from structured data, not rendered markdown)
    band = "High" if prob_pct >= 80 else "Moderate–high" if prob_pct >= 60 else "Intermediate" if prob_pct >= 40 else "Low"
    if is_fav:
        impression = (
            f"This patient's clinical profile is associated with a <strong>{prob_pct}% estimated "
            f"probability of seizure freedom at 12 months</strong> ({band} likelihood), exceeding "
            f"the model decision threshold of {threshold_pct}%. The estimate favours a positive "
            f"treatment response; it reflects population-level associations and should be "
            f"interpreted alongside the full clinical picture, treatment adherence, and comorbidity "
            f"context. Rule-based clinical adjustments have been applied where applicable."
        )
    else:
        impression = (
            f"This patient's clinical profile is associated with a <strong>{prob_pct}% estimated "
            f"probability of seizure freedom at 12 months</strong> ({band} likelihood), below the "
            f"model decision threshold of {threshold_pct}%. The profile suggests suboptimal "
            f"treatment response is likely without further management review. Consider assessing "
            f"current ASM adequacy (dose, adherence, drug-level monitoring) and whether additional "
            f"workup (video-EEG monitoring, high-resolution MRI) or tertiary epilepsy referral is "
            f"clinically indicated."
        )

    # ── SHAP two-column factors ───────────────────────────────────────────
    shap      = out.get("shap", {})
    shap_ok   = shap.get("ok", False)
    shap_doc  = shap.get("doctor", {}) if shap_ok else {}
    supports  = shap_doc.get("supports", [])[:4]
    against   = shap_doc.get("against", [])[:4]

    factors_html = ""
    if supports or against:
        sup_col = ""
        if supports:
            items = "".join(f'<li><span class="fi fi-up">▲</span>{html.escape(s)}</li>' for s in supports)
            sup_col = f'<div class="fcol fcol-sup"><div class="fcol-hdr">Factors favouring seizure freedom</div><ul class="flist">{items}</ul></div>'
        opp_col = ""
        if against:
            items = "".join(f'<li><span class="fi fi-dn">▼</span>{html.escape(a)}</li>' for a in against)
            opp_col = f'<div class="fcol fcol-opp"><div class="fcol-hdr">Factors limiting seizure control</div><ul class="flist">{items}</ul></div>'
        factors_html = f'<div class="factors-grid">{sup_col}{opp_col}</div>'

    # ── ASM notes panel ───────────────────────────────────────────────────
    asm_notes   = ml.get("rule_reasons_raw", [])
    asm_panel   = ""
    if asm_notes:
        items = "".join(f'<li>{html.escape(n)}</li>' for n in asm_notes[:3])
        asm_panel = f'<div class="cpanel cpanel-asm"><div class="cpanel-hdr">&#9888; Current ASM — Clinical Considerations</div><ul class="cpanel-list">{items}</ul></div>'

    # ── flags panel ───────────────────────────────────────────────────────
    flags_raw   = ml.get("reliability_flags", [])
    flags_panel = ""
    if flags_raw:
        items = "".join(f'<li>{html.escape(f)}</li>' for f in flags_raw[:3])
        flags_panel = f'<div class="cpanel cpanel-flags"><div class="cpanel-hdr">&#9888; Data Quality / Applicability Flags</div><ul class="cpanel-list">{items}</ul></div>'

    # ── recommended next steps ────────────────────────────────────────────
    steps = _build_next_steps(out, ranking_out, patient)
    steps_items = "".join(f'<li>{s}</li>' for s in steps)
    steps_panel = f'<div class="cpanel cpanel-steps"><div class="cpanel-hdr">&#10003; Recommended Next Steps</div><ul class="cpanel-list">{steps_items}</ul></div>'

    # ── ranking section ───────────────────────────────────────────────────
    ranking_html_inner = ""
    if ranking_out and ranking_out.get("rankings"):
        rankings  = ranking_out["rankings"]
        prob_base = float(ranking_out.get("prob_base", prob_model))
        rk_app    = int(ranking_out.get("applicability_indicator", applicability))
        rk_flags  = ranking_out.get("reliability_flags", [])

        BADGE_CFG = {
            "preferred":  {"label": "PREFERRED",  "bg": "#16a34a", "fg": "#fff", "entry_bg": "rgba(240,253,244,0.7)",  "bar": "#22c55e", "border": "#bbf7d0"},
            "acceptable": {"label": "ACCEPTABLE", "bg": "#1d4ed8", "fg": "#fff", "entry_bg": "rgba(239,246,255,0.7)",  "bar": "#3b82f6", "border": "#bfdbfe"},
            "caution":    {"label": "CAUTION",    "bg": "#b45309", "fg": "#fff", "entry_bg": "rgba(255,251,235,0.8)",  "bar": "#f59e0b", "border": "#fde68a"},
            "avoid":      {"label": "AVOID",      "bg": "#b91c1c", "fg": "#fff", "entry_bg": "rgba(254,242,242,0.8)",  "bar": "#ef4444", "border": "#fecaca"},
        }
        RANK_COLORS = {0: "#ca8a04", 1: "#64748b", 2: "#92400e"}
        RANK_SYMS   = {0: "①", 1: "②", 2: "③", 3: "④", 4: "⑤", 5: "⑥"}

        rk_app_dots = "".join(
            f'<span class="ap-dot ap-dot-{"on" if j < rk_app else "off"}" style="background:{"#38bdf8" if j < rk_app else "rgba(255,255,255,.2)"}"></span>'
            for j in range(5)
        )
        rk_app_lbl = app_lbl_map.get(min(rk_app, 4), "Very low")

        entries_html = ""
        for i, entry in enumerate(rankings):
            asm_name  = html.escape(str(entry.get("asm", "")))
            prob_adj  = float(entry.get("prob_adjusted", 0.0))
            penalty   = float(entry.get("penalty", 0.0))
            suit      = str(entry.get("suitability", "acceptable")).lower()
            cfg       = BADGE_CFG.get(suit, BADGE_CFG["acceptable"])
            c_notes   = entry.get("caution_notes") or entry.get("rule_notes", [])
            b_notes   = entry.get("benefit_notes", [])
            tier      = html.escape(str(entry.get("tier", "")))
            spectrum  = html.escape(str(entry.get("spectrum", "")))
            terat     = html.escape(str(entry.get("teratogenic_risk", "")))
            monitor   = html.escape(str(entry.get("monitoring", "")))
            ep        = round(prob_adj * 100)
            rk_color  = RANK_COLORS.get(i, "#64748b")
            rk_sym    = RANK_SYMS.get(i, f"#{i+1}")

            d_abs = abs(penalty) * 100
            if penalty < -0.005:
                delta = f'<span class="delta delta-pos">&#9650; +{d_abs:.0f}% benefit</span>'
            elif penalty > 0.005:
                delta = f'<span class="delta delta-neg">&#9660; &minus;{d_abs:.0f}% penalty</span>'
            else:
                delta = '<span class="delta delta-neutral">No rule adjustment</span>'

            tox_c = "#b91c1c" if "high" in terat.lower() else "#b45309" if "moderate" in terat.lower() else "#16a34a"
            chips = ""
            if tier:    chips += f'<span class="chip chip-tier">{tier}</span>'
            if spectrum:
                chips += f'<span class="chip {"chip-broad" if "broad" in spectrum.lower() else "chip-focal"}">{spectrum}</span>'
            if terat:   chips += f'<span class="chip chip-tox" style="border-color:{tox_c};color:{tox_c};">Teratogenic: {terat}</span>'
            if monitor: chips += f'<span class="chip chip-mon">Monitor: {monitor}</span>'

            notes = ""
            for n in b_notes:
                notes += f'<div class="rnote rnote-b">&#10003; {html.escape(n)}</div>'
            for n in c_notes:
                notes += f'<div class="rnote rnote-c">&#9888; {html.escape(n)}</div>'

            entries_html += f"""<div class="rk-entry" style="background:{cfg['entry_bg']};border-color:{cfg['border']};">
  <div class="rk-top">
    <span class="rk-rank" style="color:{rk_color};">{rk_sym}</span>
    <div class="rk-mid">
      <div class="rk-name">{asm_name}</div>
      <div class="rk-chips">{chips}</div>
    </div>
    <div class="rk-right">
      <span class="suit-badge" style="background:{cfg['bg']};color:{cfg['fg']};">{cfg['label']}</span>
      <span class="ep-num" style="color:{cfg['bar']};">{ep}%</span>
      {delta}
    </div>
  </div>
  <div class="bar-track"><div class="bar-fill" style="width:{min(ep,100)}%;background:{cfg['bar']};"></div></div>
  {f'<div class="notes-wrap">{notes}</div>' if notes else ""}
</div>"""

        rk_flags_html = ""
        if rk_flags:
            fi = "".join(f'<li>{html.escape(f)}</li>' for f in rk_flags[:3])
            rk_flags_html = f'<div class="rk-flags-blk"><strong>&#9888; Data flags:</strong><ul>{fi}</ul></div>'

        ranking_html_inner = f"""<div class="rk-section">
  <div class="rk-hdr">
    <div>
      <div class="rk-hdr-title">&#9670; Rule-Aware ASM Suitability Ranking</div>
      <div class="rk-hdr-sub">Evidence-based clinical rules applied · ILAE 2022 guidelines · Base probability: <strong>{round(prob_base*100)}%</strong></div>
    </div>
    <div class="rk-hdr-right">
      <div class="rk-app-row"><span style="font-size:.75rem;color:rgba(255,255,255,.65);margin-right:5px;">Applicability:</span>{rk_app_dots}<span style="font-size:.75rem;color:rgba(255,255,255,.7);margin-left:4px;">{rk_app_lbl}</span></div>
      <div class="legend"><span class="leg" style="background:#16a34a;">PREFERRED</span><span class="leg" style="background:#1d4ed8;">ACCEPTABLE</span><span class="leg" style="background:#b45309;">CAUTION</span><span class="leg" style="background:#b91c1c;">AVOID</span></div>
    </div>
  </div>
  <div class="rk-entries">{entries_html}</div>
  {rk_flags_html}
  <div class="rk-disc">This ranking is for clinical decision support only and does not constitute prescribing advice. Always correlate with the full clinical assessment and applicable local guidelines.</div>
</div>"""

    # ── assemble & return ─────────────────────────────────────────────────
    return dedent(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0;}}
html,body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;background:transparent;font-size:14px;}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;box-shadow:0 6px 30px rgba(15,23,42,0.10);}}

/* ── banner ── */
.banner{{padding:18px 22px 16px;position:relative;border-bottom:1px solid #f1f5f9;}}
.banner-bar{{position:absolute;top:0;left:0;right:0;height:4px;background:{banner_bar};}}
.banner-inner{{display:flex;align-items:center;gap:18px;}}
.banner-left{{flex:1;}}
.banner-label{{font-size:0.68rem;font-weight:800;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;}}
.banner-title{{font-size:1.22rem;font-weight:900;color:#0f172a;line-height:1.2;margin-bottom:8px;}}
.banner-badge{{display:inline-flex;align-items:center;padding:4px 12px;border-radius:999px;font-weight:700;font-size:0.80rem;background:{badge_bg};color:{badge_fg};border:1px solid {banner_bar}33;}}
.banner-ring{{flex-shrink:0;display:flex;flex-direction:column;align-items:center;gap:4px;}}
.ring-label{{font-size:0.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.06em;text-align:center;}}

/* ── metrics ── */
.metrics-row{{display:flex;gap:0;border-bottom:1px solid #f1f5f9;}}
.metric{{flex:1;padding:12px 16px;border-right:1px solid #f1f5f9;}}
.metric:last-child{{border-right:none;}}
.metric-lbl{{font-size:0.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;}}
.metric-val{{font-size:1.05rem;font-weight:900;color:#0f172a;}}
.metric-sub{{font-size:0.72rem;color:#64748b;margin-top:2px;}}
.ap-dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:2px;}}
.ap-dot-on{{background:#3b82f6;}}
.ap-dot-off{{background:#e2e8f0;}}
.pill{{display:inline-block;font-size:0.68rem;font-weight:700;padding:1px 7px;border-radius:4px;margin-left:5px;vertical-align:middle;}}
.pill-pos{{background:#dcfce7;color:#15803d;}}
.pill-neg{{background:#fee2e2;color:#991b1b;}}
.pill-neutral{{background:#f1f5f9;color:#64748b;}}

/* ── clinical section ── */
.clin-section{{padding:16px 20px 4px;}}
.clin-label{{font-size:0.68rem;font-weight:800;color:#94a3b8;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}}
.impression{{font-size:0.88rem;color:#1e293b;line-height:1.6;margin-bottom:14px;}}

/* ── SHAP factors grid ── */
.factors-grid{{display:flex;gap:10px;margin-bottom:12px;}}
.fcol{{flex:1;border-radius:8px;padding:10px 12px;}}
.fcol-sup{{background:#f0fdf4;border:1px solid #bbf7d0;}}
.fcol-opp{{background:#fff1f2;border:1px solid #fecdd3;}}
.fcol-hdr{{font-size:0.70rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;}}
.fcol-sup .fcol-hdr{{color:#15803d;}}
.fcol-opp .fcol-hdr{{color:#be123c;}}
.flist{{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:4px;}}
.flist li{{font-size:0.80rem;line-height:1.4;display:flex;gap:5px;}}
.fi{{font-size:0.70rem;flex-shrink:0;margin-top:2px;font-weight:900;}}
.fi-up{{color:#16a34a;}}
.fi-dn{{color:#dc2626;}}

/* ── clinical panels ── */
.cpanel{{margin:0 0 10px;border-radius:8px;padding:10px 14px 8px;}}
.cpanel-hdr{{font-size:0.70rem;font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;}}
.cpanel-list{{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:4px;}}
.cpanel-list li{{font-size:0.82rem;color:inherit;line-height:1.45;padding-left:12px;position:relative;}}
.cpanel-list li::before{{content:"›";position:absolute;left:0;font-weight:900;}}
.cpanel-asm{{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;}}
.cpanel-asm .cpanel-hdr{{color:#b45309;}}
.cpanel-asm .cpanel-list li{{color:#78350f;}}
.cpanel-flags{{background:#fefce8;border:1px solid #fef08a;border-left:4px solid #ca8a04;}}
.cpanel-flags .cpanel-hdr{{color:#92400e;}}
.cpanel-steps{{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;}}
.cpanel-steps .cpanel-hdr{{color:#1d4ed8;}}
.cpanel-steps .cpanel-list li{{color:#1e3a5f;}}

/* ── ranking section ── */
.rk-section{{border-top:3px solid #e2e8f0;}}
.rk-hdr{{padding:14px 20px 12px;background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:#fff;display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;}}
.rk-hdr-title{{font-size:1.0rem;font-weight:900;margin-bottom:3px;letter-spacing:-.01em;}}
.rk-hdr-sub{{font-size:0.75rem;color:rgba(255,255,255,0.65);}}
.rk-hdr-right{{display:flex;flex-direction:column;align-items:flex-end;gap:6px;}}
.rk-app-row{{display:flex;align-items:center;}}
.legend{{display:flex;gap:4px;flex-wrap:wrap;}}
.leg{{font-size:0.62rem;font-weight:700;padding:2px 7px;border-radius:3px;color:#fff;letter-spacing:.04em;}}
.rk-entries{{padding:10px 14px;display:flex;flex-direction:column;gap:8px;}}
.rk-entry{{border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px 10px;}}
.rk-top{{display:flex;align-items:flex-start;gap:10px;}}
.rk-rank{{font-size:1.5rem;font-weight:900;min-width:28px;line-height:1;padding-top:2px;flex-shrink:0;}}
.rk-mid{{flex:1;min-width:0;}}
.rk-name{{font-size:1.0rem;font-weight:900;color:#0f172a;margin-bottom:5px;}}
.rk-chips{{display:flex;gap:4px;flex-wrap:wrap;}}
.chip{{font-size:0.62rem;font-weight:600;padding:2px 7px;border-radius:4px;border:1px solid;}}
.chip-tier{{background:#f0f9ff;border-color:#7dd3fc;color:#0369a1;}}
.chip-broad{{background:#f0fdf4;border-color:#86efac;color:#15803d;}}
.chip-focal{{background:#fefce8;border-color:#fde047;color:#854d0e;}}
.chip-tox{{background:#fff7ed;font-size:0.60rem;}}
.chip-mon{{background:#faf5ff;border-color:#d8b4fe;color:#6b21a8;font-size:0.60rem;}}
.rk-right{{display:flex;flex-direction:column;align-items:flex-end;gap:4px;min-width:115px;}}
.suit-badge{{font-size:0.62rem;font-weight:800;padding:3px 9px;border-radius:4px;letter-spacing:.06em;}}
.ep-num{{font-size:1.5rem;font-weight:900;line-height:1;}}
.delta{{font-size:0.73rem;font-weight:600;}}
.delta-pos{{color:#16a34a;}}
.delta-neg{{color:#b91c1c;}}
.delta-neutral{{color:#94a3b8;}}
.bar-track{{height:6px;background:#e2e8f0;border-radius:999px;margin:8px 0 6px;overflow:hidden;}}
.bar-fill{{height:100%;border-radius:999px;}}
.notes-wrap{{display:flex;flex-direction:column;gap:3px;margin-top:4px;}}
.rnote{{font-size:0.78rem;padding:4px 10px;border-radius:5px;line-height:1.35;}}
.rnote-b{{background:rgba(240,253,244,0.9);color:#166534;border-left:3px solid #4ade80;}}
.rnote-c{{background:rgba(255,251,235,0.9);color:#78350f;border-left:3px solid #fbbf24;}}
.rk-flags-blk{{margin:0 14px 8px;padding:8px 12px;background:#fffbeb;border:1px solid #fde68a;border-radius:7px;font-size:0.78rem;color:#92400e;}}
.rk-flags-blk ul{{margin:4px 0 0 14px;}}
.rk-flags-blk li{{margin:2px 0;}}
.rk-disc{{padding:8px 20px 12px;font-size:0.70rem;color:#94a3b8;border-top:1px solid #f1f5f9;margin-top:2px;line-height:1.55;}}
</style></head>
<body>
<div class="card">

  <!-- BANNER -->
  <div class="banner">
    <div class="banner-bar"></div>
    <div class="banner-inner">
      <div class="banner-left">
        <div class="banner-label">Treatment Response Prediction &nbsp;·&nbsp; 12-Month Outcome</div>
        <div class="banner-title">{"Seizure-Free at 12 Months" if is_fav else "Not Seizure-Free at 12 Months"}</div>
        <span class="banner-badge">{badge_text}</span>
      </div>
      <div class="banner-ring">
        {ring_svg}
        <div class="ring-label">Seizure-free<br>probability</div>
      </div>
    </div>
  </div>

  <!-- METRICS -->
  <div class="metrics-row">
    <div class="metric">
      <div class="metric-lbl">Adjusted Probability</div>
      <div class="metric-val">{prob_pct}% {radj}</div>
      <div class="metric-sub">Base: {round(prob_model*100)}%</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">Model Applicability</div>
      <div class="metric-val">{''.join(f'<span class="ap-dot ap-dot-{"on" if j < applicability else "off"}"></span>' for j in range(5))} {app_lbl}</div>
    </div>
    <div class="metric">
      <div class="metric-lbl">Prediction Model</div>
      <div class="metric-val" style="font-size:0.82rem;padding-top:3px;">{model_name}</div>
    </div>
  </div>

  <!-- CLINICAL ASSESSMENT -->
  <div class="clin-section">
    <div class="clin-label">&#9670; Clinical Assessment</div>
    <div class="impression">{impression}</div>
    {factors_html}
    {asm_panel}
    {flags_panel}
    {steps_panel}
  </div>

  {ranking_html_inner}
</div>
</body></html>""").strip()


def render():
    _init_state()

    MODEL_PATH = "models/asm_response_prediction.pkl"

    # ASM options — hardcoded fallback since model is on backend
    trained_asms = []

    # CSS/theme - keep in main page (not iframe)
    st.markdown(
        dedent("""
        <style>
        .block-container{
            padding-top:0rem !important;
        }

        section.main > div{
            padding-top:0rem !important;
        }
          .asm-title { font-size: 2.5rem; font-weight: 800; color:#1E3A5F; }
          .asm-subtitle { color:#49576B; margin-top:-0.4rem; }
          .asm-section-title { font-size: 1.25rem; font-weight: 800; color:#1E3A5F; margin-bottom:0.25rem; }
          .asm-divider { height: 1px; background: rgba(226,232,240,0.9); margin: 0.9rem 0; }
          div[data-testid="column"] { padding-left: 1rem; padding-right: 1rem; }
          div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius: 12px !important; }
          div[data-testid="stForm"]{ border: 0 !important; outline: 0 !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; margin: 0 !important; }
          input::placeholder { color: #9CA3AF !important; opacity: 1 !important; }
          ul[role="listbox"] li:first-child { color: #9CA3AF !important; }

          div[data-testid="stFormSubmitButton"]{
            display:flex !important;
            justify-content:center !important;
            align-items:center !important;
            width:100% !important;
            margin-top: 2rem !important;
          }

          div[data-testid="stFormSubmitButton"] button,
          button[data-testid="stBaseButton-primary"],
          button[kind="primary"] {
            width: auto !important;
            padding: 0.80rem 3.2rem !important;
            border-radius: 50px !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.03em !important;
            background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%) !important;
            border: none !important;
            color: #ffffff !important;
            box-shadow: 0 4px 15px rgba(0, 91, 234, 0.3) !important;
            white-space: nowrap !important;
            text-align: center !important;
            transition: all 0.18s ease !important;
          }

          div[data-testid="stFormSubmitButton"] button:hover,
          button[data-testid="stBaseButton-primary"]:hover,
          button[kind="primary"]:hover {
            background: linear-gradient(90deg, #00c6fb 0%, #005bea 100%) !important;
            box-shadow: 0 6px 20px rgba(0, 91, 234, 0.4) !important;
            color: #ffffff !important;
            transform: translateY(-2px) !important;
          }
        </style>
        """).strip(),
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 40, 1], vertical_alignment="top")

    with mid:
        st.markdown(
            dedent("""
            <div>
              <div class="asm-title">Anti-Seizure Medication Response</div>
              <p class="asm-subtitle">Enter key patient, clinical, and investigation data to estimate seizure freedom at 12 months.</p>
            </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        req = " *" if st.session_state.asm_show_required else ""
        SELECT = "Select an option"

        sex_opts = [SELECT, "Female", "Male"]
        seizure_type_opts = [SELECT, "Focal", "Generalized", "Mixed"]
        mri_lesion_opts = [SELECT, "Hippocampal_Sclerosis", "Tumor", "Cortical_Dysplasia"]
        eeg_opts = [SELECT, "Generalized", "Focal", "Normal", "Multifocal"]
        yn_opts = [SELECT, "Yes", "No"]

        trained_asms = []
        if trained_asms:
            asm_opts = [SELECT] + sorted(list(set(map(str, trained_asms))))
        else:
            asm_opts = [
                SELECT, "Levetiracetam", "Lamotrigine", "Valproate", "Phenobarbital",
                "Carbamazepine", "Phenytoin"
            ]

        with st.form("asm_form_ui_only", clear_on_submit=False):
            st.markdown('<div class="asm-section-title">Clinical Profile</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)

            with c1:
                st.number_input(
                    f"Age (years){req}", min_value=0, max_value=120,
                    value=st.session_state.age, placeholder="Enter age", key="age"
                )
                st.selectbox(
                    f"Seizure Type{req}", seizure_type_opts,
                    index=_safe_index(seizure_type_opts, st.session_state.seizure_type),
                    placeholder="Select seizure type", key="seizure_type"
                )
                st.number_input(
                    f"Number of Seizures Before ASM Initiation{req}",
                    min_value=0, max_value=50,
                    value=st.session_state.pretreatment_seizure_count,
                    placeholder="Enter number of seizures", key="pretreatment_seizure_count"
                )

            with c2:
                st.selectbox(
                    f"Biological Sex (at birth){req}", sex_opts,
                    index=_safe_index(sex_opts, st.session_state.sex),
                    placeholder="Select biological sex", key="sex"
                )
                st.selectbox(
                    "MRI Lesion Type (if available)", mri_lesion_opts,
                    index=_safe_index(mri_lesion_opts, st.session_state.mri_lesion_type),
                    placeholder="Select MRI lesion type", key="mri_lesion_type"
                )
                st.number_input(
                    "Number of Prior Anti-Seizure Medications Tried",
                    min_value=0, max_value=50,
                    value=st.session_state.prior_asm_exposure_count,
                    placeholder="Enter number of prior ASMs", key="prior_asm_exposure_count"
                )

            with c3:
                st.number_input(
                    f"Age at Seizure Onset (years){req}", min_value=0, max_value=120,
                    value=st.session_state.age_of_onset,
                    placeholder="Enter age at seizure onset", key="age_of_onset"
                )
                st.selectbox(
                    "EEG Findings (interictal status)", eeg_opts,
                    index=_safe_index(eeg_opts, st.session_state.eeg_status_detail),
                    placeholder="Select EEG findings", key="eeg_status_detail"
                )
                st.selectbox(
                    f"Current Anti-Seizure Medication{req}", asm_opts,
                    index=_safe_index(asm_opts, st.session_state.current_asm),
                    placeholder="Select current ASM", key="current_asm"
                )

            st.markdown('<div class="asm-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="asm-section-title">Associated Conditions</div>', unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1:
                st.selectbox("Psychiatric Disorders (diagnosed)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.psychiatric_disorder),
                             placeholder="Select status", key="psychiatric_disorder")
                st.selectbox("History of Head Trauma (with loss of consciousness)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.head_trauma),
                             placeholder="Select status", key="head_trauma")
                st.selectbox("Family History of Epilepsy", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.family_history),
                             placeholder="Select status", key="family_history")

            with d2:
                st.selectbox("Intellectual Disability (clinical diagnosis)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.intellectual_disability),
                             placeholder="Select status", key="intellectual_disability")
                st.selectbox("Central Nervous System Infection (e.g., meningitis, encephalitis)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.cns_infection),
                             placeholder="Select status", key="cns_infection")

            with d3:
                st.selectbox("Cerebrovascular Disease (e.g., stroke, hemorrhage)", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.cerebrovascular_disease),
                             placeholder="Select status", key="cerebrovascular_disease")
                st.selectbox("Substance or Alcohol Use Disorder", yn_opts,
                             index=_safe_index(yn_opts, st.session_state.substance_alcohol_abuse),
                             placeholder="Select status", key="substance_alcohol_abuse")

            submitted = st.form_submit_button("Predict Treatment Response", use_container_width=False)

        if submitted:
            age = st.session_state.age
            onset = st.session_state.age_of_onset
            if age is not None and onset is not None and onset > age:
                st.warning("Age of onset cannot be greater than current age.")
                return

            sample_patient = {
                "age": _clean_num(st.session_state.age),
                "age_of_onset": _clean_num(st.session_state.age_of_onset),
                "pretreatment_seizure_count": _clean_num(st.session_state.pretreatment_seizure_count),
                "prior_asm_exposure_count": _clean_num(st.session_state.prior_asm_exposure_count),

                "sex": _clean_cat_keep_placeholder(st.session_state.sex),
                "seizure_type": _clean_cat_keep_placeholder(st.session_state.seizure_type),
                "current_asm": _clean_cat_keep_placeholder(st.session_state.current_asm),

                "mri_lesion_type": _clean_cat_keep_placeholder(st.session_state.mri_lesion_type),
                "eeg_status_detail": _clean_cat_keep_placeholder(st.session_state.eeg_status_detail),

                "psychiatric_disorder": _clean_cat_keep_placeholder(st.session_state.psychiatric_disorder),
                "intellectual_disability": _clean_cat_keep_placeholder(st.session_state.intellectual_disability),
                "cerebrovascular_disease": _clean_cat_keep_placeholder(st.session_state.cerebrovascular_disease),
                "head_trauma": _clean_cat_keep_placeholder(st.session_state.head_trauma),
                "cns_infection": _clean_cat_keep_placeholder(st.session_state.cns_infection),
                "substance_alcohol_abuse": _clean_cat_keep_placeholder(st.session_state.substance_alcohol_abuse),
                "family_history": _clean_cat_keep_placeholder(st.session_state.family_history),
            }

            required_keys = ["age", "age_of_onset", "pretreatment_seizure_count", "sex", "seizure_type", "current_asm"]
            missing = [k for k in required_keys if _is_missing_required(sample_patient.get(k)) or sample_patient.get(k) is None]
            if missing:
                st.warning(f"Please fill/select required fields: {', '.join(missing)}")
                return

            try:
                out = post("/epilepsy_diagnosis/asm/predict", sample_patient)
            except APIError as e:
                st.error(f"Prediction failed: {e}")
                return
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

            # Also fetch ASM ranking
            ranking_out = None
            try:
                ranking_out = post("/epilepsy_diagnosis/asm/rank", sample_patient)
            except Exception:
                ranking_out = None

            combined_html = _render_combined_card(out, ranking_out, sample_patient)

            # ── iframe height estimation ──────────────────────────────────────
            # card_base: banner(122) + metrics(70) + clin-header(44) + impression(130) + buffer(34) = 400
            # steps_h computed from actual step count (each step ≈ 3 lines × 19px + 4px gap = ~61px)
            card_base   = 400
            n_steps     = len(_build_next_steps(out, ranking_out, sample_patient))
            steps_h     = 48 + n_steps * 61          # 48px frame/header + 61px per item

            shap_ok     = (out.get("shap") or {}).get("ok", False)
            shap_add    = 145 if shap_ok else 0

            asm_notes_len = len((out.get("ml_details") or {}).get("rule_reasons_raw", []))
            flags_raw_len = len((out.get("ml_details") or {}).get("reliability_flags", []))
            asm_add   = (65 + min(asm_notes_len, 3) * 17) if asm_notes_len else 0
            flags_add = (65 + min(flags_raw_len, 3) * 17) if flags_raw_len else 0

            # Ranking: 95px header/footer + 130px per ASM entry + 26px per note
            rk_rows = len(ranking_out["rankings"]) if ranking_out and ranking_out.get("rankings") else 0
            rk_notes = sum(
                len(e.get("caution_notes") or e.get("rule_notes", [])) +
                len(e.get("benefit_notes", []))
                for e in (ranking_out.get("rankings", []) if ranking_out else [])
            )
            rk_section = (95 + rk_rows * 130 + rk_notes * 26) if rk_rows else 0

            total_height = card_base + steps_h + shap_add + asm_add + flags_add + rk_section
            total_height = max(total_height, 600)

            components.html(combined_html, height=total_height, scrolling=False)
