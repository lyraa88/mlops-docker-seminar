import os, time, sys, datetime, torch
from zoneinfo import ZoneInfo

sys.path.append("/app")
from mnist_model import retrain_once

SHARE = os.environ.get("SHARE_DIR", "/app/share_storage")

RUN_AT_HH = int(os.environ.get("RUN_AT_HH", "11"))
RUN_AT_MM = int(os.environ.get("RUN_AT_MM", "50"))
TZ_NAME   = os.environ.get("TZ", "Asia/Seoul")

CHECK_INTERVAL = 30
LAST_RUN_FLAG = os.path.join(SHARE, "metrics", "last_daily_run.txt")

def already_ran_today(now_local: datetime.datetime) -> bool:
    if not os.path.exists(LAST_RUN_FLAG): return False
    try:
        y, m, d = map(int, open(LAST_RUN_FLAG).read().strip().split("-"))
        return (now_local.year, now_local.month, now_local.day) == (y, m, d)
    except: return False

def mark_ran_today(now_local: datetime.datetime):
    os.makedirs(os.path.dirname(LAST_RUN_FLAG), exist_ok=True)
    with open(LAST_RUN_FLAG, "w") as f:
        f.write(f"{now_local.year:04d}-{now_local.month:02d}-{now_local.day:02d}")

def main():
    tz = ZoneInfo(TZ_NAME)
    print(f"[trigger] daily schedule {RUN_AT_HH:02d}:{RUN_AT_MM:02d} ({TZ_NAME})")
    while True:
        try:
            now = datetime.datetime.now(tz)
            if (now.hour == RUN_AT_HH and now.minute == RUN_AT_MM) and not already_ran_today(now):
                print("[trigger] time reached, starting retrain...")
                retrain_once(SHARE, epochs=2, lr=1e-3)
                mark_ran_today(now)
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print("[trigger] error:", e)
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

    