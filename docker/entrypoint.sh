#!/bin/bash
set -e

echo "==> Waiting for MySQL..."
python - <<'PY'
import os, time
import MySQLdb

host = os.environ.get("MYSQL_HOST", "db")
port = int(os.environ.get("MYSQL_PORT", "3306"))
user = os.environ.get("MYSQL_USER", "root")
password = os.environ.get("MYSQL_PASSWORD", "")
db = os.environ.get("MYSQL_DATABASE", "quexian")

for i in range(60):
    try:
        conn = MySQLdb.connect(host=host, port=port, user=user, passwd=password, db=db, connect_timeout=2)
        conn.close()
        print("MySQL is ready.")
        break
    except Exception as e:
        time.sleep(2)
else:
    raise RuntimeError("MySQL not ready after waiting.")
PY

echo "==> Migrate..."
python manage.py migrate --noinput

echo "==> Collect static..."
python manage.py collectstatic --noinput

echo "==> Start Gunicorn..."
# 你项目wsgi位置按实际修改：quexian.wsgi:application
exec gunicorn quexian.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --threads 2 \
  --timeout 300
