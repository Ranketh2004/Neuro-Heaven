import streamlit as st
from utils.api_client import post, get, APIError

def _set_page(slug: str):
    st.query_params["page"] = slug
    st.rerun()

def open(mode: str = "signin"):
    st.session_state["auth_open"] = True
    st.session_state["auth_mode"] = mode

def close():
    st.session_state["auth_open"] = False

@st.dialog(" ", width="small")
def render_dialog():
    st.markdown(
        r"""
        <style>
        /* -----------------------------
           1) REMOVE STREAMLIT HEADER BAR
        ------------------------------*/
        [data-testid="stDialogHeader"]{
            display:none !important;
            height:0 !important;
            padding:0 !important;
            margin:0 !important;
        }

        /* Some Streamlit builds also show a native close icon button */
        [data-testid="stDialog"] button[aria-label="Close"],
        [data-testid="stDialog"] button[title="Close"],
        [data-testid="stDialog"] button[aria-label="close"],
        [data-testid="stDialog"] button[title="close"]{
            display:none !important;
            visibility:hidden !important;
        }

        /* Make the dialog background transparent; we draw our own card */
        div[data-testid="stDialog"] > div{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* -----------------------------
           2) CENTER WRAPPER + CARD
        ------------------------------*/
        .auth-wrap{
            width: 100%;
            display:flex;
            justify-content:center;
            align-items:center;
            padding: 14px 10px;
        }
        .auth-card{
            width: 520px;
            max-width: 92vw;
            background: #ffffff;
            border-radius: 22px;
            border: 1px solid rgba(226,232,240,0.9);
            box-shadow: 0 28px 70px rgba(15, 52, 96, 0.18);
            padding: 26px 26px 22px 26px;
            position: relative;
        }

        /* -----------------------------
           3) CLOSE BUTTON TOP-RIGHT
        ------------------------------*/
        
        .auth-card div[data-testid="stButton"]:first-of-type button{
            position:absolute !important;
            top: 14px;
            right: 14px;
            width: 36px !important;
            height: 36px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(226,232,240,0.95) !important;
            background: #ffffff !important;
            box-shadow: none !important;
            color: #94a3b8 !important;
            font-size: 20px !important;
            font-weight: 900 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
        .auth-card div[data-testid="stButton"]:first-of-type button:hover{
            background:#f8fafc !important;
            color:#64748b !important;
        }

        /* -----------------------------
           4) TITLE + SUBTITLE (CENTERED)
        ------------------------------*/
        .auth-title{
            text-align:center;
            font-size: 34px;
            font-weight: 800;
            color:#0f172a;
            margin: 8px 0 8px 0;
        }
        .auth-sub{
            text-align:center;
            color:#64748b;
            font-size: 16px;
            margin: 0 0 18px 0;
        }

        /* -----------------------------
           5) INPUTS WITH ICONS
        ------------------------------*/
        /* Hide ONLY text input label, keep checkbox label */
        .auth-card div[data-testid="stTextInput"] label{
            display:none !important;
        }

        .auth-card div[data-testid="stTextInput"] input{
            border-radius: 12px !important;
            border: 1px solid rgba(203,213,225,0.95) !important;
            padding: 12px 14px !important;
            padding-left: 46px !important;     /* room for left icon */
            padding-right: 46px !important;    /* room for eye icon */
            font-size: 16px !important;
            background:#ffffff !important;
        }
        .auth-card div[data-testid="stTextInput"] input:focus{
            border-color: rgba(2,132,199,0.75) !important;
            box-shadow: 0 0 0 4px rgba(2,132,199,0.15) !important;
            outline: none !important;
        }

        /* Email icon */
        .auth-card input[aria-label="Email"]{
            background-repeat:no-repeat !important;
            background-position: 14px 50% !important;
            background-size:18px 18px !important;
            background-image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2394a3b8'>\
<path d='M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4-8 5-8-5V6l8 5 8-5v2z'/>\
</svg>");
        }

        /* Lock icon for Password */
        .auth-card input[aria-label="Password"]{
            background-repeat:no-repeat !important;
            background-position: 14px 50% !important;
            background-size:18px 18px !important;
            background-image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2394a3b8'>\
<path d='M12 17a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm6-7h-1V8a5 5 0 0 0-10 0v2H6c-1.1 0-2 .9-2 2v8c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-8c0-1.1-.9-2-2-2zm-3 0H9V8a3 3 0 0 1 6 0v2z'/>\
</svg>");
        }

        /* Tighter widget spacing like screenshot */
        .auth-card .stTextInput{ margin-top: 12px !important; margin-bottom: 0 !important; }
        .auth-card .stCheckbox{ margin-top: 10px !important; margin-bottom: 0 !important; }

        /* Checkbox label visible */
        .auth-card div[data-testid="stCheckbox"] label{
            font-size: 15px !important;
            color:#475569 !important;
            font-weight: 500 !important;
        }

        /* Remember + forgot row */
        .auth-row-right{
            text-align:right;
            padding-top: 10px;
        }
        .auth-row-right a{
            color:#0284c7;
            font-weight: 700;
            text-decoration:none;
        }
        .auth-row-right a:hover{ text-decoration:underline; }

        /* -----------------------------
           6) PRIMARY BUTTON FULL WIDTH
        ------------------------------*/
        /* First primary action button after close button */
        .auth-card div[data-testid="stButton"]:not(:first-of-type) button[kind="primary"]{
            width:100% !important;
            border-radius: 12px !important;
            padding: 14px 14px !important;
            font-weight: 800 !important;
            font-size: 16px !important;
            background: #0284c7 !important;
            border: none !important;
            margin-top: 10px !important;
        }
        .auth-card div[data-testid="stButton"]:not(:first-of-type) button[kind="primary"]:hover{
            background:#0272aa !important;
        }

        /* -----------------------------
           7) DIVIDER + SOCIAL BUTTONS
        ------------------------------*/
        .auth-divider{
            display:flex;
            align-items:center;
            gap: 12px;
            margin: 18px 0 14px 0;
            color:#94a3b8;
            font-weight: 600;
            justify-content:center;
        }
        .auth-divider:before,.auth-divider:after{
            content:"";
            flex:1;
            height:1px;
            background: rgba(226,232,240,1);
        }
        .auth-social{
            display:flex;
            gap:14px;
        }
        .auth-social a{
            flex:1;
            display:flex;
            align-items:center;
            justify-content:center;
            gap:10px;
            border: 1px solid rgba(203,213,225,0.9);
            border-radius: 12px;
            padding: 12px;
            background:#fff;
            font-weight: 700;
            color:#0f172a;
            text-decoration:none;
        }
        .auth-social a:hover{ background:#f8fafc; }

        /* style ONLY our close button */
        button[kind="secondary"]{
            width: 36px !important;
            height: 36px !important;
            border-radius: 10px !important;
            border: 1px solid rgba(226,232,240,0.95) !important;
            background: #ffffff !important;
            box-shadow: none !important;
            color: #94a3b8 !important;
            font-size: 20px !important;
            font-weight: 900 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
        button[kind="secondary"]:hover{
            background:#f8fafc !important;
            color:#64748b !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    mode = st.session_state.get("auth_mode", "signin")

    spacer, btn_col = st.columns([0.92, 0.08])
    with btn_col:
        if st.button("✕", key="auth_close_btn"):
            close()
            st.rerun()

    if mode == "signin":
        st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Enter your details to access your account</div>', unsafe_allow_html=True)

        email = st.text_input("Email", key="li_email", placeholder="name@example.com", label_visibility="collapsed")
        password = st.text_input("Password", type="password", key="li_pw", placeholder="Password", label_visibility="collapsed")

        c1, c2 = st.columns([1, 1])
        with c1:
            remember = st.checkbox("Remember me", value=False, key="li_remember")
        with c2:
            st.markdown(
                '<div class="auth-row-right"><a href="#" style="outline:none;" onclick="return false;">Forgot password?</a></div>',
                unsafe_allow_html=True
            )

        ok = st.button("Sign In", type="primary", use_container_width=True)

        if ok:
            try:
                data = post("/auth/login", {"email": email, "password": password})
                st.session_state["token"] = data["access_token"]
                me = get("/auth/me", token=st.session_state["token"])
                st.session_state["user"] = me
                target = st.session_state.pop("pending_page", "home")
                close()
                _set_page(target)
            except APIError as e:
                st.error(str(e))

        st.markdown('<div class="auth-divider">Or continue with</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="auth-social">
                <a href="#" onclick="return false;">
                    <img src="https://www.gstatic.com/images/branding/product/1x/googleg_48dp.png" width="18" height="18">
                    Google
                </a>
                <a href="#" onclick="return false;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="18" height="18">
                    GitHub
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown('<div class="auth-title">Create an account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Join today and start your journey</div>', unsafe_allow_html=True)

        full_name = st.text_input("Full Name", key="su_name", placeholder="Full Name", label_visibility="collapsed")
        email2 = st.text_input("Email", key="su_email", placeholder="name@example.com", label_visibility="collapsed")
        pw1 = st.text_input("Password", type="password", key="su_pw1", placeholder="Password", label_visibility="collapsed")
        agree = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="su_agree")

        ok2 = st.button("Create Account", type="primary", use_container_width=True)

        if ok2:
            if not agree:
                st.error("You must agree to continue.")
            elif len(pw1) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                try:
                    post("/auth/signup", {"full_name": full_name, "email": email2, "password": pw1})
                    st.success("Account created. Now sign in.")
                    st.session_state["auth_mode"] = "signin"
                    st.rerun()
                except APIError as e:
                    st.error(str(e))

        st.markdown('<div class="auth-divider">Or continue with</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="auth-social">
                <a href="#" onclick="return false;">
                    <img src="https://www.gstatic.com/images/branding/product/1x/googleg_48dp.png" width="18" height="18">
                    Google
                </a>
                <a href="#" onclick="return false;">
                    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="18" height="18">
                    GitHub
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)
