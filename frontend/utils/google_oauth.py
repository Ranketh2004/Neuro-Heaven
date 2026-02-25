import os
import json
import requests
from pathlib import Path
from urllib.parse import urlencode, parse_qs
from urllib.request import urlopen

def _load_repo_env():
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass

    try:
        text = env_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            # Avoid overwriting existing env vars
            if k and v and os.getenv(k) is None:
                os.environ[k] = v
    except Exception:
        pass

_load_repo_env()


class GoogleOAuth:
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3000")
        
        self.auth_uri = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_uri = "https://oauth2.googleapis.com/token"
        self.userinfo_uri = "https://www.googleapis.com/oauth2/v2/userinfo"
        
    def get_authorization_url(self, state: str = "security_token") -> str:
        """
        Generate the Google OAuth authorization URL
        """
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "consent"
        }
        return f"{self.auth_uri}?{urlencode(params)}"
    
    def exchange_code_for_token(self, code: str) -> dict:
        """
        Exchange authorization code for access token
        """
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        
        response = requests.post(self.token_uri, data=payload)
        response.raise_for_status()
        return response.json()
    
    def get_user_info(self, access_token: str) -> dict:
        """
        Fetch user info using access token
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(self.userinfo_uri, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def handle_callback(self, code: str) -> dict:
        """
        Complete OAuth flow: exchange code for token and get user info
        Returns: {"success": bool, "user": dict or None, "error": str or None}
        """
        try:
            # Exchange code for tokens
            token_response = self.exchange_code_for_token(code)
            access_token = token_response.get("access_token")
            
            if not access_token:
                return {"success": False, "user": None, "error": "Failed to get access token"}
            
            # Get user info
            user_info = self.get_user_info(access_token)
            
            return {
                "success": True,
                "user": {
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "picture": user_info.get("picture"),
                    "google_id": user_info.get("id")
                },
                "error": None,
                "token": token_response  # Store full token response if needed
            }
        except Exception as e:
            return {"success": False, "user": None, "error": str(e)}
