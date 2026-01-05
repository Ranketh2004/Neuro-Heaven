# views/home.py
import streamlit as st

HOME_HTML = """
<div id="home" class="nh-hero-wrapper">
<div class="nh-hero">
<div class="nh-pill">
<div class="nh-pill-icon">✓</div>
<span>Multimodal AI for Clinical Decision Support</span>
</div>

<h1 class="nh-hero-title">
<span>Advanced AI for Epilepsy</span>
<span>Diagnosis &amp; Treatment</span>
</h1>

<p class="nh-hero-subtext">
Combining cutting-edge machine learning with multimodal medical data
to provide comprehensive clinical decision support for epilepsy
diagnosis, seizure localization, and personalized treatment planning.
</p>
</div>
</div>

<!-- STATS SECTION -->
<div id="stats" class="nh-stats-wrapper">
<div class="nh-stats">

<div class="nh-stat-card">
<div class="nh-stat-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E3A5F" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<line x1="12" y1="20" x2="12" y2="10"></line>
<line x1="18" y1="20" x2="18" y2="4"></line>
<line x1="6" y1="20" x2="6" y2="14"></line>
</svg>
</div>
<div class="nh-stat-value">10,000+</div>
<div class="nh-stat-label-main">Patients Analyzed</div>
</div>

<div class="nh-stat-card">
<div class="nh-stat-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E3A5F" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"></circle>
<circle cx="12" cy="12" r="4"></circle>
<circle cx="12" cy="12" r="1"></circle>
</svg>
</div>
<div class="nh-stat-value">94%</div>
<div class="nh-stat-label-main">Diagnostic Accuracy</div>
</div>

<div class="nh-stat-card">
<div class="nh-stat-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E3A5F" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polyline>
</svg>
</div>
<div class="nh-stat-value">5 min</div>
<div class="nh-stat-label-main">Average Analysis Time</div>
</div>

<div class="nh-stat-card">
<div class="nh-stat-icon">
<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#1E3A5F" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V6l-8-4-8 4v6c0 6 8 10 8 10z"></path>
</svg>
</div>
<div class="nh-stat-value">HIPAA</div>
<div class="nh-stat-label-main">Compliant &amp; Secure</div>
</div>

</div>
</div>

<!-- WHY NEUROHEAVEN SECTION -->
<div id="why" class="nh-why-section">
<div class="nh-why-inner">
<div class="nh-why-header">
<h2 class="nh-why-title">Why NeuroHeaven?</h2>
<p class="nh-why-subtitle">
Empowering clinicians with AI tools that enhance diagnostic accuracy and treatment outcomes
</p>
</div>

<div class="nh-why-grid">

<div class="nh-benefit-card">
<div class="nh-benefit-icon">
<!-- Zap icon -->
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#74B0D3" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polyline>
</svg>
</div>
<h3 class="nh-benefit-title">Faster Diagnosis</h3>
<p class="nh-benefit-text">
Reduce diagnostic time from weeks to minutes with AI-powered analysis.
</p>
</div>

<div class="nh-benefit-card">
<div class="nh-benefit-icon">
<!-- Shield icon -->
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#74B0D3" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M12 22s8-4 8-10V6l-8-4-8 4v6c0 6 8 10 8 10z"></path>
</svg>
</div>
<h3 class="nh-benefit-title">Clinical Accuracy</h3>
<p class="nh-benefit-text">
Validated algorithms trained on extensive medical datasets.
</p>
</div>

<div class="nh-benefit-card">
<div class="nh-benefit-icon">
<!-- Target icon -->
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#74B0D3" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"></circle>
<circle cx="12" cy="12" r="4"></circle>
<circle cx="12" cy="12" r="1"></circle>
</svg>
</div>
<h3 class="nh-benefit-title">Personalized Care</h3>
<p class="nh-benefit-text">
Tailored treatment recommendations based on individual patient profiles.
</p>
</div>

<div class="nh-benefit-card">
<div class="nh-benefit-icon">
<!-- Users icon -->
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#74B0D3" stroke-width="2"
stroke-linecap="round" stroke-linejoin="round">
<path d="M17 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"></path>
<circle cx="9" cy="7" r="4"></circle>
<path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
<path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
</svg>
</div>
<h3 class="nh-benefit-title">Expert Support</h3>
<p class="nh-benefit-text">
Augment clinical decision-making with interpretable AI insights.
</p>
</div>

</div>
</div>
</div>
"""

def render():
    st.markdown(HOME_HTML, unsafe_allow_html=True)
