import streamlit as st
from datetime import datetime

def _format_date(value) -> str:
    """Format a date value (ISO string or datetime) to a readable string."""
    if not value:
        return "N/A"
    try:
        if isinstance(value, datetime):
            return value.strftime("%B %d, %Y")
        # Try ISO string parsing
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except Exception:
        return str(value)[:10]  # fallback: first 10 chars (date part)


def render():
    user: dict = st.session_state.get("user") or {}

    full_name = user.get("full_name") or user.get("name") or "User"
    email = user.get("email", "")
    picture = user.get("picture")
    google_id = user.get("google_id")
    is_google = bool(google_id or picture)
    member_since = _format_date(user.get("created_at"))

    initials = "".join(p[0].upper() for p in full_name.split() if p)[:2] or "U"

    if picture:
        avatar_html = f'<img class="pf-avatar-img" src="{picture}" alt="Profile photo" />'
    else:
        avatar_html = f'<div class="pf-avatar-initials">{initials}</div>'

    auth_badge = (
        '<span class="pf-badge pf-badge--google">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">'
        '<path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>'
        '<path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>'
        '<path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>'
        '<path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>'
        "</svg>"
        " Google account</span>"
    )
    email_badge = (
        '<span class="pf-badge pf-badge--email">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="flex-shrink:0">'
        '<path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>'
        '<path d="m22 7-10 7L2 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
        " Email account</span>"
    )

    st.markdown(
        """
<style>
.pf-wrapper {
    max-width: 500px !important;
    margin: 2rem auto 0 auto;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.pf-card {
    background: #ffffff;
    border: 1px solid #E5EDF7;
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(15, 52, 96, 0.07);
    overflow: hidden;
}
.pf-header {
    background: linear-gradient(135deg, #20A0D8 0%, #0E5C7A 100%);
    padding: 2rem 1.75rem 1.25rem 1.75rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}
.pf-avatar-img {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    object-fit: cover;
    border: 4px solid rgba(255,255,255,0.45);
    box-shadow: 0 4px 20px rgba(0,0,0,0.18);
}
.pf-avatar-initials {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: rgba(255,255,255,0.22);
    border: 4px solid rgba(255,255,255,0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 900;
    font-size: 2rem;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 20px rgba(0,0,0,0.18);
}
.pf-header-name {
    color: #ffffff;
    font-weight: 800;
    font-size: 1.45rem;
    text-align: center;
    line-height: 1.2;
}
.pf-header-email {
    color: rgba(255,255,255,0.80);
    font-size: 0.92rem;
    text-align: center;
}
.pf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pf-badge--google {
    background: rgba(255,255,255,0.18);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.3);
}
.pf-badge--email {
    background: rgba(255,255,255,0.18);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.3);
}
.pf-body {
    padding: 1.5rem 1.75rem 2rem 1.75rem;
}
.pf-section-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}
.pf-field-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid #F1F5F9;
}
.pf-field-row:last-child {
    border-bottom: none;
}
.pf-field-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: #F3F7FF;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #20A0D8;
}
.pf-field-content {
    flex: 1;
    min-width: 0;
}
.pf-field-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 0.2rem;
}
.pf-field-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1E3A5F;
    word-break: break-word;
}
.pf-logout-wrap {
    max-width: 350px;
    margin: 1rem auto 2rem auto;
}
.pf-logout-wrap [data-testid="stButton"] button {
    background: linear-gradient(135deg, #ff4b4b 0%, #cc2200 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(204, 34, 0, 0.3) !important;
    transition: all 0.15s ease !important;
}
.pf-logout-wrap [data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(204, 34, 0, 0.45) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 8, 1])
    with col:
        st.markdown(
        f"""
<div class="pf-card">
<div class="pf-header">
{avatar_html}
<div>
<div class="pf-header-name">{full_name}</div>
{'<div class="pf-header-email">' + email + "</div>" if email else ""}
<div style="margin-top:0.65rem;text-align:center;">
{auth_badge if is_google else email_badge}
</div>
</div>
</div>
<div class="pf-body">
<div class="pf-section-label">Account Details</div>

<div class="pf-field-row">
<div class="pf-field-icon">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2"/>
<path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>
</div>
<div class="pf-field-content">
<div class="pf-field-label">Full Name</div>
<div class="pf-field-value">{full_name}</div>
</div>
</div>

<div class="pf-field-row">
<div class="pf-field-icon">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
<path d="M20 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
<path d="m22 7-10 7L2 7" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>
<div class="pf-field-content">
<div class="pf-field-label">Email Address</div>
<div class="pf-field-value">{email or "N/A"}</div>
</div>
</div>

<div class="pf-field-row">
<div class="pf-field-icon">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
<rect x="3" y="4" width="18" height="18" rx="2" ry="2" stroke="currentColor" stroke-width="2"/>
<line x1="16" y1="2" x2="16" y2="6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<line x1="8" y1="2" x2="8" y2="6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
<line x1="3" y1="10" x2="21" y2="10" stroke="currentColor" stroke-width="2"/>
</svg>
</div>
<div class="pf-field-content">
<div class="pf-field-label">Member Since</div>
<div class="pf-field-value">{member_since}</div>
</div>
</div>

<div class="pf-field-row">
<div class="pf-field-icon">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
</svg>
</div>
<div class="pf-field-content">
<div class="pf-field-label">Sign-in Method</div>
<div class="pf-field-value">{"Google OAuth" if is_google else "Email & Password"}</div>
</div>
</div>

</div>
</div>
""",
        unsafe_allow_html=True,
        )
