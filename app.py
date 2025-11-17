from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd, time, uuid, os, random
from datetime import datetime
import joblib

APP_TITLE = "Tone Judgment Study"
STUDY_CSV = "a2_study_with_outputs.csv"
MODEL_PKL = "a2_model.pkl"
LOG_DIR = "logs"
LOG_TRIALS = os.path.join(LOG_DIR, "a3_logs.csv")
LOG_SESS = os.path.join(LOG_DIR, "a3_sessions.csv")

os.makedirs(LOG_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace_with_random_secret"

# load data and model
if not os.path.exists(STUDY_CSV):
    raise FileNotFoundError("Put a2_study_with_outputs.csv next to app.py")
df = pd.read_csv(STUDY_CSV)

pipe = None
if os.path.exists(MODEL_PKL):
    try:
        pipe = joblib.load(MODEL_PKL)
    except Exception:
        pipe = None

need_cols = ["input_text","ground_truth_label","model_output_label","model_score","human_rating_placeholder","ai_readable_label","ai_rationale"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column {c} in {STUDY_CSV}")

def sample_trials(n=18):
    n = max(1, min(int(n), len(df)))
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    return idxs[:n]

def survey_code():
    return "VS-A3-" + uuid.uuid4().hex[:6].upper()

def write_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(h,"")) for h in header) + "\n")

@app.route("/", methods=["GET","POST"])
def home():
    # allow MTurk auto fill via URL query params
    q = request.args
    preset_pid = q.get("pid","")
    preset_cond = q.get("cond","")
    preset_n = q.get("n","18")
    preset_sync = q.get("sync","0")

    if request.method == "POST":
        session["pid"] = request.form.get("pid","anon").strip() or "anon"
        cond_raw = request.form.get("cond","noai")
        session["cond"] = "ai" if cond_raw == "ai" else "noai"
        session["sync"] = request.form.get("sync","0") == "1"
        n_trials = int(request.form.get("n","18"))
        session["idxs"] = sample_trials(n_trials)
        session["pos"] = 0
        session["start_ts"] = time.time()
        session["trial_start"] = time.time()
        session["sid"] = uuid.uuid4().hex[:12]
        session["survey_done"] = False
        # instructions step
        return redirect(url_for("consent"))
    return render_template("home.html",
                           title=APP_TITLE,
                           preset_pid=preset_pid,
                           preset_cond=preset_cond,
                           preset_n=preset_n,
                           preset_sync=preset_sync)

@app.route("/consent", methods=["GET","POST"])
def consent():
    if "idxs" not in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        if request.form.get("agree","off") == "on":
            return redirect(url_for("task"))
    return render_template("consent.html", title=APP_TITLE)

@app.route("/task", methods=["GET","POST"])
def task():
    if "idxs" not in session:
        return redirect(url_for("home"))

    pos = session.get("pos",0)
    idxs = session.get("idxs",[])
    total = len(idxs)
    if pos >= total:
        return redirect(url_for("survey"))

    # handle previous answer and log
    if request.method == "POST":
        ans = request.form.get("answer","")
        rt = time.time() - float(session.get("trial_start", time.time()))
        prev_idx = idxs[pos-1] if pos > 0 else idxs[0]
        rprev = df.iloc[prev_idx]
        header = [
            "sid","pid","cond","sync_ai","trial_index","message_index",
            "input_text","ai_label","ai_score","ai_rationale",
            "answer","response_time_ms","ts","user_agent"
        ]
        row = {
            "sid": session["sid"],
            "pid": session["pid"],
            "cond": session["cond"],
            "sync_ai": int(session.get("sync", False)),
            "trial_index": pos,
            "message_index": prev_idx,
            "input_text": str(rprev["input_text"]).replace(","," "),
            "ai_label": rprev["ai_readable_label"] if session["cond"]=="ai" else "",
            "ai_score": rprev["model_score"] if session["cond"]=="ai" else "",
            "ai_rationale": rprev["ai_rationale"] if session["cond"]=="ai" else "",
            "answer": ans,
            "response_time_ms": int(rt*1000),
            "ts": datetime.utcnow().isoformat(),
            "user_agent": request.headers.get("User-Agent","").replace(","," ")
        }
        write_csv(LOG_TRIALS, row, header)
        session["pos"] = pos + 1
        pos = session["pos"]
        if pos >= total:
            return redirect(url_for("survey"))

    # next item
    idx = idxs[pos]
    row = df.iloc[idx]
    session["trial_start"] = time.time()

    message = row["input_text"]
    ai_info = None
    if session["cond"] == "ai":
        if session.get("sync", False) and pipe is not None:
            try:
                prob = float(pipe.predict_proba([message])[:,1][0])
                label = "CONSENSUAL_FLIRT" if prob >= 0.5 else "TOO_FORWARD"
                ai_info = {"label": label, "score": prob, "rationale": row.get("ai_rationale","lexical pattern features")}
            except Exception:
                ai_info = {"label": row["ai_readable_label"], "score": row["model_score"], "rationale": row["ai_rationale"]}
        else:
            ai_info = {"label": row["ai_readable_label"], "score": row["model_score"], "rationale": row["ai_rationale"]}

    return render_template("task.html",
                           title=APP_TITLE,
                           pos=pos+1, total=len(idxs),
                           message=message,
                           cond=session["cond"],
                           ai_info=ai_info)

@app.route("/survey", methods=["GET","POST"])
def survey():
    if "idxs" not in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        conf = request.form.get("confidence","3")
        code = survey_code()
        header = ["sid","pid","cond","sync_ai","n_trials","study_minutes","confidence_1to5","code","ts"]
        row = {
            "sid": session["sid"],
            "pid": session["pid"],
            "cond": session["cond"],
            "sync_ai": int(session.get("sync", False)),
            "n_trials": len(session["idxs"]),
            "study_minutes": round((time.time() - session["start_ts"])/60.0, 2),
            "confidence_1to5": conf,
            "code": code,
            "ts": datetime.utcnow().isoformat()
        }
        write_csv(LOG_SESS, row, header)
        session["survey_done"] = True
        session["code"] = code
        return redirect(url_for("finish"))
    return render_template("survey.html", title=APP_TITLE)

@app.route("/finish")
def finish():
    if not session.get("survey_done", False):
        return redirect(url_for("home"))
    return render_template("finish.html", title=APP_TITLE, code=session.get("code","VS-A3-XXXXXX"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
