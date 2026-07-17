"""Daily report via email (SMTP) — HTML formatted."""
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

log = logging.getLogger(__name__)


def _send_email(subject: str, html: str, plain: str) -> None:
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    to_addr = os.environ["NOTIFY_EMAIL"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(host, port) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)
    log.info("Email report sent to %s", to_addr)


def _pill(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600">{text}</span>'
    )


def _ticker_chips(tickers: list[str], color: str) -> str:
    if not tickers:
        return '<span style="color:#888;font-style:italic">None</span>'
    chips = "".join(
        f'<span style="display:inline-block;background:{color}18;color:{color};'
        f'border:1px solid {color}44;padding:3px 10px;border-radius:6px;'
        f'margin:3px 4px 3px 0;font-size:14px;font-weight:600">{t}</span>'
        for t in tickers
    )
    return chips


def _build_html(report: dict[str, Any]) -> str:
    date = report.get("date", "N/A")
    pv = report.get("portfolio_value", 0)
    analyzed = report.get("tickers_analyzed", 0)
    placed = report.get("orders_placed", 0)
    top_longs = report.get("top_longs", [])
    top_shorts = report.get("top_shorts", [])

    orders_color = "#22c55e" if placed > 0 else "#64748b"
    orders_pill = _pill(f"{placed} Orders", orders_color)

    long_chips = _ticker_chips(top_longs, "#22c55e")
    short_chips = _ticker_chips(top_shorts, "#ef4444")

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0f172a;padding:32px 16px">
<tr><td align="center">
<table width="560" cellpadding="0" cellspacing="0" style="max-width:560px;width:100%">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#1e3a5f 0%,#0f2744 100%);border-radius:12px 12px 0 0;padding:28px 32px">
    <div style="font-size:11px;letter-spacing:3px;text-transform:uppercase;color:#60a5fa;margin-bottom:6px">Brain Topology Trader</div>
    <div style="font-size:22px;font-weight:700;color:#f1f5f9">Daily Report</div>
    <div style="font-size:13px;color:#94a3b8;margin-top:4px">{date}</div>
  </td></tr>

  <!-- Stats row -->
  <tr><td style="background:#1e293b;padding:0">
    <table width="100%" cellpadding="0" cellspacing="0">
      <tr>
        <td width="33%" style="padding:20px 24px;border-right:1px solid #334155;text-align:center">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Portfolio Value</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9">${pv:,.0f}</div>
        </td>
        <td width="33%" style="padding:20px 24px;border-right:1px solid #334155;text-align:center">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Tickers Scanned</div>
          <div style="font-size:22px;font-weight:700;color:#f1f5f9">{analyzed}</div>
        </td>
        <td width="33%" style="padding:20px 24px;text-align:center">
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Orders Placed</div>
          <div style="font-size:22px;font-weight:700;color:{orders_color}">{placed}</div>
        </td>
      </tr>
    </table>
  </td></tr>

  <!-- Long signals -->
  <tr><td style="background:#162032;padding:20px 32px;border-top:1px solid #1e3a5f">
    <div style="font-size:11px;color:#22c55e;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:10px">
      &#9650; Long Signals
    </div>
    <div>{long_chips}</div>
  </td></tr>

  <!-- Short signals -->
  <tr><td style="background:#1a1a2e;padding:20px 32px;border-top:1px solid #1e3a5f">
    <div style="font-size:11px;color:#ef4444;text-transform:uppercase;letter-spacing:2px;font-weight:600;margin-bottom:10px">
      &#9660; Short Signals
    </div>
    <div>{short_chips}</div>
  </td></tr>

  <!-- Footer -->
  <tr><td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;border-top:1px solid #1e293b">
    <div style="font-size:11px;color:#334155;text-align:center">
      NCP v5 Ensemble &nbsp;·&nbsp; Brain Topology Trader &nbsp;·&nbsp; Educational use only
    </div>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def _build_plain(report: dict[str, Any]) -> str:
    date = report.get("date", "N/A")
    pv = report.get("portfolio_value", 0)
    analyzed = report.get("tickers_analyzed", 0)
    placed = report.get("orders_placed", 0)
    top_longs = report.get("top_longs", [])
    top_shorts = report.get("top_shorts", [])
    return (
        f"Brain Topology Trader — {date}\n"
        f"Portfolio: ${pv:,.0f}  |  Scanned: {analyzed}  |  Orders: {placed}\n"
        f"\nLongs:  {', '.join(top_longs) or 'None'}\n"
        f"Shorts: {', '.join(top_shorts) or 'None'}\n"
    )


def send_daily_report(report: dict[str, Any]) -> None:
    date = report.get("date", "N/A")
    subject = f"Brain Topology Trader — {date}"
    html = _build_html(report)
    plain = _build_plain(report)

    if os.environ.get("SMTP_USER") and os.environ.get("NOTIFY_EMAIL"):
        try:
            _send_email(subject, html, plain)
        except Exception as exc:
            log.error("Email failed: %s", exc)
    else:
        log.warning("No email credentials — report logged only")
        log.info("REPORT:\n%s", plain)
