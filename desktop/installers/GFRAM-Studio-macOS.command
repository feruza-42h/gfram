#!/bin/zsh
# =====================================================================
#  GFRAM Studio — o'rnatish va ishga tushirish / установка и запуск (macOS)
#  Ikki marta bosing / дважды щёлкните. Keyingi safar shunchaki ochiladi.
# =====================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
APP_HOME="${GFRAM_APP_HOME:-$HOME/.gfram/studio-app}"
ENV="$APP_HOME/env"
LAUNCHER="${GFRAM_LAUNCHER:-$HOME/Desktop/GFRAM Studio.command}"
PY_VER="3.12.10"
PY_PKG_URL="https://www.python.org/ftp/python/${PY_VER}/python-${PY_VER}-macos11.pkg"

step() { print -P "\n%F{cyan}%B==> $1%b%f"; }
ok()   { print -P "%F{green}✔ $1%f"; }
die()  {
  print -P "\n%F{red}%B✘ $1%b%f"
  osascript -e "display alert \"GFRAM Studio\" message \"$1\" as critical" >/dev/null 2>&1
  read -k1 "?Davom etish uchun istalgan tugma / нажмите любую клавишу…"; exit 1
}

clear
print -P "%F{blue}%B   GFRAM Studio%b%f — gibrid yuz tanish tizimi / гибридное распознавание лиц\n"

# ---------- already installed: just start (updates are done inside the app)
if [[ -x "$ENV/bin/gfram-studio" && "${1:-}" != "--reinstall" ]]; then
  step "Ishga tushirilmoqda / запуск…"
  exec "$ENV/bin/gfram-studio"
fi

# ---------- 1. Python 3.10–3.14
find_python() {
  local p v
  for p in python3.12 python3.13 python3.11 python3.10 python3.14 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
           /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    command -v "$p" >/dev/null 2>&1 || [[ -x "$p" ]] || continue
    v=$("$p" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    case "$v" in 3.10|3.11|3.12|3.13|3.14) print -r -- "$p"; return 0;; esac
  done
  return 1
}

step "1/4  Python tekshirilmoqda / проверка Python"
PY="$(find_python)"
if [[ -z "$PY" ]]; then
  print "Python 3.10+ topilmadi. python.org dan Python ${PY_VER} o'rnatiladi."
  print "Python 3.10+ не найден. Будет установлен Python ${PY_VER} с python.org."
  TMP_PKG="$(mktemp -d)/python.pkg"
  curl -fL --progress-bar "$PY_PKG_URL" -o "$TMP_PKG" || die "Python yuklab olinmadi / не удалось скачать Python. Internetni tekshiring."
  print "Administrator paroli so'raladi / будет запрошен пароль администратора…"
  osascript -e "do shell script \"installer -pkg '$TMP_PKG' -target /\" with administrator privileges" \
    || die "Python o'rnatilmadi / Python не установлен."
  PY="$(find_python)" || die "Python o'rnatildi, lekin topilmadi / Python установлен, но не найден."
fi
ok "Python: $PY ($("$PY" -V 2>&1))"

# ---------- 2. Private environment + GFRAM
step "2/4  GFRAM Studio o'rnatilmoqda (birinchi marta 3–10 daqiqa) / установка (впервые 3–10 минут)"
mkdir -p "$APP_HOME"
"$PY" -m venv "$ENV" || die "Muhit yaratilmadi / не удалось создать окружение."
"$ENV/bin/python" -m pip install --upgrade --disable-pip-version-check pip >/dev/null
WHEEL=( "$HERE"/gfram-*.whl(N) )
if (( ${#WHEEL} )); then
  SPEC="${WHEEL[-1]}[studio]"; print "Mahalliy paket / локальный пакет: ${WHEEL[-1]:t}"
else
  SPEC="gfram[studio]>=3.4.0"
fi
"$ENV/bin/python" -m pip install --upgrade --disable-pip-version-check "$SPEC" \
  || die "O'rnatish bajarilmadi / установка не выполнена. Internetni tekshiring."
ok "GFRAM $("$ENV/bin/python" -c 'import gfram;print(gfram.__version__)')"

step "Modellar tayyorlanmoqda / подготовка моделей…"
"$ENV/bin/python" - <<'PY' || die "Modellar yuklanmadi / модели не загружены."
import logging; logging.disable(logging.WARNING)
from gfram.api.simple_recognizer import SimpleRecognizer
import tempfile
r = SimpleRecognizer(db_path=tempfile.mkdtemp(), contribute=False)
_ = r.hybrid          # geometric model + MobileFaceNet (downloaded once, checksum-verified)
print('models ready:', r.embedder.version, r.hybrid.fusion.appearance_model)
PY

# ---------- 3. Camera permission
step "3/4  Kameraga ruxsat / доступ к камере"
print "macOS so'raganda «Ruxsat berish / Разрешить» ni bosing."
"$ENV/bin/python" - <<'PY'
import time, cv2
cap = cv2.VideoCapture(0)
t = time.time(); ok = False
while time.time() - t < 6:
    ok, _ = cap.read()
    if ok: break
    time.sleep(0.2)
cap.release()
print('camera:', 'OK' if ok else 'ruxsat yo‘q yoki kamera band / нет доступа или камера занята')
PY

# ---------- 4. Desktop launcher
step "4/4  Ish stolida yorliq / ярлык на рабочем столе"
cat > "$LAUNCHER" <<SH
#!/bin/zsh
exec "$ENV/bin/gfram-studio"
SH
chmod +x "$LAUNCHER"
ok "$LAUNCHER"

[[ -n "${GFRAM_NO_LAUNCH:-}" ]] && exit 0
step "GFRAM Studio ishga tushirilmoqda / запуск GFRAM Studio"
exec "$ENV/bin/gfram-studio"
