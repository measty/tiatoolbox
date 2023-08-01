"""

Bokeh server authentication hooks are building blocks that can be used by
experienced users to implement any authentication flow they require. This
example is a "toy" example that is only intended to demonstrate how those
building blocks fit together. It should not be used as-is for "production"
use. Users looking for pre-built auth flows that work out of the box should
consider a higher level tool, such as Panel:

    https://panel.holoviz.org/user_guide/Authentication.html

"""
import json
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import tornado
from tornado.web import RequestHandler


# could define get_user_async instead
def get_user(request_handler):
    return request_handler.get_cookie("user")


# could also define get_login_url function (but must give up LoginHandler)
login_url = "/login"


# optional login page for login_url
class LoginHandler(RequestHandler):
    def get(self):
        try:
            errormessage = self.get_argument("error")
        except Exception:
            errormessage = ""
        # import pdb; pdb.set_trace()
        next_url = self.get_argument("next", "/bokeh_app")
        self.next_url = next_url
        # parse text in next_url to get the 'demo' query parameter
        parsed = urlparse(next_url)
        query = parse_qs(parsed.query)
        demo = query.get("demo", [None])[0]

        needs_auth = False
        demo_path = Path(f"/app_data/{demo}/overlays")
        config_file = list(demo_path.glob("*config.json"))
        print(config_file)
        if len(config_file) > 0:
            with open(config_file[0]) as f:
                config = json.load(f)
                if "password" in config:
                    needs_auth = True

        if not needs_auth:
            # skip login
            self.set_current_user("default", expires=20)
            self.redirect(next_url)
            return

        self.render(
            "login.html", errormessage=errormessage, action=f"/login?next={next_url}"
        )

    def check_permission(self, username, password, config):
        # !!!
        # !!! This code below is a toy demonstration of the API, and not
        # !!! intended for "real" use. A real app should use these APIs
        # !!! to connect Oauth or some other established auth workflow.
        # !!!
        if password == config["password"]:
            return True
        return False

    def post(self):
        # import pdb; pdb.set_trace()
        username = self.get_argument("username", secrets.token_urlsafe(16))
        password = self.get_argument("password", "")

        next_url = self.get_argument("next", "/bokeh_app")
        self.next_url = next_url
        # parse text in next_url to get the 'demo' query parameter
        parsed = urlparse(next_url)
        query = parse_qs(parsed.query)
        demo = query.get("demo", [None])[0]

        demo_path = Path(f"/app_data/{demo}/overlays")
        config_file = list(demo_path.glob("*config.json"))
        if len(config_file) > 0:
            with open(config_file[0]) as f:
                config = json.load(f)

        auth = self.check_permission(username, password, config)
        if auth:
            self.set_current_user(username, 20)
            next_url = self.get_argument("next", "/bokeh_app")
            self.redirect(next_url)
        else:
            error_msg = "?error=" + tornado.escape.url_escape("Login incorrect")
            self.redirect(login_url + error_msg)

    def set_current_user(self, user, expires):
        if user:
            self.set_cookie("user", tornado.escape.json_encode(user), max_age=expires)
        else:
            self.clear_cookie("user")


# optional logout_url, available as curdoc().session_context.logout_url
logout_url = "/logout"


# optional logout handler for logout_url
class LogoutHandler(RequestHandler):
    def get(self):
        self.clear_cookie("user")
        self.redirect("/")
