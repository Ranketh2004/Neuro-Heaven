"""
API Testing Guide for Authentication Endpoints

This file demonstrates how to test the authentication endpoints.
Run from backend directory: python test/test_auth_api.py
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.OKBLUE}ℹ {msg}{Colors.ENDC}")


def print_request(method, path, data=None):
    print(f"{Colors.OKCYAN}{method} {path}{Colors.ENDC}")
    if data:
        print(f"Body: {json.dumps(data, indent=2)}")


def test_signup():
    """Test user registration endpoint."""
    print_section("TEST 1: User Registration (Sign Up)")
    
    path = "/auth/signup"
    payload = {
        "full_name": "Test User",
        "email": f"testuser_{int(time.time())}@example.com",
        "password": "SecurePassword123"
    }
    
    print_request("POST", f"{API_BASE}{path}", payload)
    print()
    
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        
        if response.status_code == 201:
            print_success("User registered successfully!")
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return data["email"], payload["password"]
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
    except Exception as e:
        print_error(f"Error: {e}")
        return None, None


def test_login(email, password):
    """Test user login endpoint."""
    print_section("TEST 2: User Login")
    
    path = "/auth/login"
    payload = {
        "email": email,
        "password": password
    }
    
    print_request("POST", f"{API_BASE}{path}", payload)
    print()
    
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        
        if response.status_code == 200:
            print_success("Login successful!")
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return data["access_token"]
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None


def test_get_current_user(token):
    """Test get current user endpoint."""
    print_section("TEST 3: Get Current User Profile")
    
    path = "/auth/me"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    print_request("GET", f"{API_BASE}{path}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print()
    
    try:
        response = requests.get(f"{API_BASE}{path}", headers=headers, timeout=10)
        
        if response.status_code == 200:
            print_success("Retrieved current user successfully!")
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_invalid_token():
    """Test with invalid token."""
    print_section("TEST 4: Invalid Token Handling")
    
    path = "/auth/me"
    headers = {
        "Authorization": "Bearer invalid_token_12345"
    }
    
    print_request("GET", f"{API_BASE}{path}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print()
    
    try:
        response = requests.get(f"{API_BASE}{path}", headers=headers, timeout=10)
        
        if response.status_code == 401:
            print_success("Correctly rejected invalid token!")
            print(f"Response: {response.json()}")
            return True
        else:
            print_error(f"Expected 401, got {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def test_missing_token():
    """Test without token."""
    print_section("TEST 5: Missing Token Handling")
    
    path = "/auth/me"
    
    print_request("GET", f"{API_BASE}{path}")
    print("Headers: (none)")
    print()
    
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=10)
        
        if response.status_code == 401:
            print_success("Correctly rejected missing token!")
            print(f"Response: {response.json()}")
            return True
        else:
            print_error(f"Expected 401, got {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}NeuroHeaven Authentication API Tests{Colors.ENDC}")
    print(f"Testing API at: {API_BASE}")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE}/docs", timeout=5)
        print_success("API server is running!")
    except:
        print_error("Cannot connect to API server. Make sure it's running:")
        print("  cd backend && python main.py")
        return
    
    # Run tests
    email, password = test_signup()
    
    if email and password:
        token = test_login(email, password)
        
        if token:
            test_get_current_user(token)
            test_invalid_token()
            test_missing_token()
            
            print_section("All Tests Completed!")
            print_success("Authentication system is working correctly!")
        else:
            print_error("Login test failed, skipping remaining tests")
    else:
        print_error("Signup test failed, skipping remaining tests")


if __name__ == "__main__":
    main()
