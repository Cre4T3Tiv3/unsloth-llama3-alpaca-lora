"""Authenticate with the Hugging Face Hub using environment variables or interactive login.

This script handles login to the Hugging Face Hub using the `HUGGINGFACE_TOKEN` environment variable
or falls back to interactive login if not found. It supports loading from a `.env` file using `dotenv`.

Typical usage:
    $ python scripts/hf_login.py
    OR
    $ make login
"""

import os

import requests
from dotenv import load_dotenv
from huggingface_hub import login, whoami


def main():
    """Performs authentication with the Hugging Face Hub.

    The function checks for `HUGGINGFACE_TOKEN` in the environment or `.env` file.
    If the token is present, it attempts non-interactive login. Otherwise, it falls back
    to interactive login via CLI. Prints the authenticated username on success.
    """
    load_dotenv()

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        print("🔐 Using HUGGINGFACE_TOKEN from environment or .env file...")
        try:
            login(token=token)
            user = whoami(token=token)["name"]
            print(f"Authenticated as {user}")
        except requests.exceptions.HTTPError:
            print("Invalid Hugging Face token. Please check and try again.")
    else:
        print("No HUGGINGFACE_TOKEN found. Falling back to interactive login...")
        try:
            login()
            user = whoami()["name"]
            print(f"Authenticated as {user}")
        except requests.exceptions.HTTPError:
            print("Login failed. Make sure you entered a valid token.")


if __name__ == "__main__":
    main()
