# Auto-loaded only when /opt/display-kv is added to PYTHONPATH by the
# experimental launcher.  Fail CLOSED: an enabled but uninstalled hook must
# terminate this Python process, never silently fall back to ordinary KV.
import os

if os.environ.get("DISPLAY_KV_ENABLED") == "1":
    try:
        from display_kv_glm import install

        install()
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(78)
