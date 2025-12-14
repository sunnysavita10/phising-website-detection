'''pip install requests beautifulsoup4 tldextract dnspython python-whois'''

'''
URL
 ↓
URL Parser + DNS + SSL + HTML Fetch
 ↓
Feature Extraction Functions
 ↓
30 Feature Dict (-1 / 0 / 1)
 ↓
ML Model
'''

import socket
import ssl
import requests
import dns.resolver
import whois

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime


class FeatureExtractor:
    """
    Extract phishing-related features from a URL in real-time.
    Output values follow dataset convention:
    1 = suspicious / phishing indicator
    0 = legitimate
    -1 = unknown / not available in real-time
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (PhishGuard-AI)"
        }

    # -------------------------
    # URL LEVEL FEATURES
    # -------------------------

    def having_ip_address(self, url: str) -> int:
        try:
            socket.inet_aton(urlparse(url).hostname)
            return 1
        except:
            return 0

    def url_length(self, url: str) -> int:
        return 1 if len(url) >= 75 else 0

    def shortening_service(self, url: str) -> int:
        shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co"]
        return 1 if any(s in url for s in shorteners) else 0

    def having_at_symbol(self, url: str) -> int:
        return 1 if "@" in url else 0

    def double_slash_redirecting(self, url: str) -> int:
        return 1 if url.rfind("//") > 7 else 0

    def prefix_suffix(self, domain: str) -> int:
        return 1 if "-" in domain else 0

    def having_sub_domain(self, domain: str) -> int:
        return 1 if domain.count(".") > 1 else 0

    # -------------------------
    # DNS / SSL / DOMAIN
    # -------------------------

    def dns_record(self, domain: str) -> int:
        try:
            dns.resolver.resolve(domain, "A")
            return 1
        except:
            return 0

    def ssl_final_state(self, url: str) -> int:
        try:
            hostname = urlparse(url).hostname
            context = ssl.create_default_context()
            with context.wrap_socket(
                socket.socket(), server_hostname=hostname
            ) as s:
                s.settimeout(3)
                s.connect((hostname, 443))
            return 1
        except:
            return 0

    def domain_registration_length(self, domain: str) -> int:
        try:
            w = whois.whois(domain)
            if w.creation_date and w.expiration_date:
                days = (w.expiration_date - w.creation_date).days
                return 1 if days >= 365 else 0
        except:
            pass
        return 0

    def age_of_domain(self, domain: str) -> int:
        try:
            w = whois.whois(domain)
            if w.creation_date:
                days = (datetime.now() - w.creation_date).days
                return 1 if days >= 180 else 0
        except:
            pass
        return 0

    # -------------------------
    # HTML BASED FEATURES
    # -------------------------

    def fetch_html(self, url: str) -> str:
        try:
            r = requests.get(
                url,
                timeout=self.timeout,
                headers=self.headers,
                allow_redirects=True,
            )
            return r.text
        except:
            return ""

    def iframe_present(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        return 1 if soup.find("iframe") else 0

    def submitting_to_email(self, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        for f in soup.find_all("form"):
            action = f.get("action", "")
            if "mailto:" in action:
                return 1
        return 0

    def abnormal_url(self, domain: str, html: str) -> int:
        return 1 if domain not in html else 0

    def request_url(self, domain: str, html: str) -> int:
        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src and domain not in src:
                return 1
        return 0

    # -------------------------
    # MAIN EXTRACTOR
    # -------------------------

    def extract(self, url: str) -> dict:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        html = self.fetch_html(url)

        features = {
            # URL structure
            "having_IP_Address": self.having_ip_address(url),
            "URL_Length": self.url_length(url),
            "Shortining_Service": self.shortening_service(url),
            "having_At_Symbol": self.having_at_symbol(url),
            "double_slash_redirecting": self.double_slash_redirecting(url),
            "Prefix_Suffix": self.prefix_suffix(domain),
            "having_Sub_Domain": self.having_sub_domain(domain),

            # Security
            "SSLfinal_State": self.ssl_final_state(url),
            "Domain_registeration_length": self.domain_registration_length(domain),
            "age_of_domain": self.age_of_domain(domain),
            "DNSRecord": self.dns_record(domain),

            # HTML behavior
            "Request_URL": self.request_url(domain, html),
            "Submitting_to_email": self.submitting_to_email(html),
            "Abnormal_URL": self.abnormal_url(domain, html),
            "Iframe": self.iframe_present(html),

            # ========= DEFAULT / UNKNOWN FEATURES =========
            "Favicon": -1,
            "port": -1,
            "HTTPS_token": -1,
            "URL_of_Anchor": -1,
            "Links_in_tags": -1,
            "SFH": -1,
            "Redirect": -1,
            "on_mouseover": -1,
            "RightClick": -1,
            "popUpWidnow": -1,
            "web_traffic": -1,
            "Page_Rank": -1,
            "Google_Index": -1,
            "Links_pointing_to_page": -1,
            "Statistical_report": -1,
        }

        return features

        
