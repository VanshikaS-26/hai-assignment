import streamlit as st
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

# ---------- Data + model loading ----------

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Put a2_study_with_outputs.csv next to app.py")
    df = pd.read_csv(path)
    need_cols = [
        "input_text", "ground_truth_label", "model_output_label",
        "model_score", "human_rating_placeholder",
        "ai_readable_label", "ai_rationale"
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {STUDY_CSV}")
    return df


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


df = load_data(STUDY_CSV)
pipe = load_model(MODEL_PKL)


# ---------- Helpers ----------

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
        f.write(",".join(str(row.get(h, "")) for h in header) + "\n")


def init_state():
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "pid" not in st.session_state:
        st.session_state.pid = ""
    if "cond" not in st.session_state:
        st.session_state.cond = "noai"
    if "sync_ai" not in st.session_state:
        st.session_state.sync_ai = False
    if "idxs" not in st.session_state:
        st.session_state.idxs = []
    if "pos" not in st.session_state:
        st.session_state.pos = 0
    if "start_ts" not in st.session_state:
        st.session_state.start_ts = None
    if "trial_start" not in st.session_state:
        st.session_state.trial_start = None
    if "sid" not in st.session_state:
        st.session_state.sid = ""
    if "survey_done" not in st.session_state:
        st.session_state.survey_done = False
    if "code" not in st.session_state:
        st.session_state.code = ""


# ---------- Pages ----------

def page_home():
    st.title(APP_TITLE)
    st.subheader("Welcome")

    with st.form("home_form"):
        pid = st.text_input("Participant ID (optional)", value=st.session_state.pid)
        cond = st.selectbox("Condition", ["noai", "ai"], index=0 if st.session_state.cond == "noai" else 1)
        n_trials = st.number_input(
            "Number of trials",
            min_value=1,
            max_value=len(df),
            value=18,
            step=1
        )
        sync_ai = st.checkbox("Use live AI model (if available)", value=st.session_state.sync_ai)

        start_btn = st.form_submit_button("Start")

    if start_btn:
        st.session_state.pid = pid.strip() or "anon"
        st.session_state.cond = cond
        st.session_state.sync_ai = sync_ai
        st.session_state.idxs = sample_trials(n_trials)
        st.session_state.pos = 0
        st.session_state.start_ts = time.time()
        st.session_state.trial_start = time.time()
        st.session_state.sid = uuid.uuid4().hex[:12]
        st.session_state.survey_done = False
        st.session_state.code = ""
        st.session_state.page = "consent"
        st.rerun()


def page_consent():
    st.title(APP_TITLE)
    st.subheader("Consent Form")

    st.write(
        "Please read the consent information above (your actual consent text for the study). "
        "If you agree to take part in this study, check the box below and continue."
    )

    with st.form("consent_form"):
        agree = st.checkbox("I have read the information and I agree to participate.")
        btn = st.form_submit_button("Continue")

    if btn:
        if not agree:
            st.warning("You need to agree to participate before continuing.")
        else:
            st.session_state.page = "task"
            st.session_state.trial_start = time.time()
            st.rerun()

    if st.button("Back to start"):
        st.session_state.page = "home"
        st.rerun()


def page_task():
    if not st.session_state.idxs:
        st.session_state.page = "home"
        st.rerun()

    pos = st.session_state.pos
    idxs = st.session_state.idxs
    total = len(idxs)

    if pos >= total:
        st.session_state.page = "survey"
        st.rerun()

    idx = idxs[pos]
    row = df.iloc[idx]

    if st.session_state.trial_start is None:
        st.session_state.trial_start = time.time()

    st.title(APP_TITLE)
    st.subheader(f"Message {pos + 1} of {total}")

    st.markdown("### Message")
    st.write(row["input_text"])

    ai_info = None
    if st.session_state.cond == "ai":
        # sync with model if enabled and model exists
        if st.session_state.sync_ai and pipe is not None:
            try:
                prob = float(pipe.predict_proba([row["input_text"]])[:, 1][0])
                label = "CONSENSUAL_FLIRT" if prob >= 0.5 else "TOO_FORWARD"
                ai_info = {
                    "label": label,
                    "score": prob,
                    "rationale": row.get("ai_rationale", "lexical pattern features")
                }
            except Exception:
                ai_info = {
                    "label": row["ai_readable_label"],
                    "score": row["model_score"],
                    "rationale": row["ai_rationale"]
                }
        else:
            ai_info = {
                "label": row["ai_readable_label"],
                "score": row["model_score"],
                "rationale": row["ai_rationale"]
            }

    if ai_info is not None:
        st.markdown("### AI Assistant Opinion")
        st.write(f"**AI label:** {ai_info['label']}")
        st.write(f"**AI confidence score:** {ai_info['score']}")
        st.write(f"**AI rationale:** {ai_info['rationale']}")

    st.markdown("### Your judgment")
    with st.form(f"trial_form_{pos}"):
        answer = st.radio(
            "How would you judge this message?",
            ["CONSENSUAL_FLIRT", "TOO_FORWARD"],
            key=f"answer_{pos}"
        )
        submit = st.form_submit_button("Submit and continue")

    if submit:
        rt = time.time() - float(st.session_state.trial_start or time.time())
        header = [
            "sid", "pid", "cond", "sync_ai", "trial_index", "message_index",
            "input_text", "ai_label", "ai_score", "ai_rationale",
            "answer", "response_time_ms", "ts", "user_agent"
        ]
        row_log = {
            "sid": st.session_state.sid,
            "pid": st.session_state.pid,
            "cond": st.session_state.cond,
            "sync_ai": int(bool(st.session_state.sync_ai)),
            "trial_index": pos,
            "message_index": idx,
            "input_text": str(row["input_text"]).replace(",", " "),
            "ai_label": ai_info["label"] if ai_info else "",
            "ai_score": ai_info["score"] if ai_info else "",
            "ai_rationale": ai_info["rationale"] if ai_info else "",
            "answer": answer,
            "response_time_ms": int(rt * 1000),
            "ts": datetime.utcnow().isoformat(),
            # Streamlit does not expose user agent easily; leave blank or constant
            "user_agent": "streamlit"
        }
        write_csv(LOG_TRIALS, row_log, header)

        st.session_state.pos += 1
        st.session_state.trial_start = time.time()
        st.rerun()


def page_survey():
    st.title(APP_TITLE)
    st.subheader("Short survey")

    n_trials = len(st.session_state.idxs)

    with st.form("survey_form"):
        conf = st.slider(
            "How confident do you feel about your judgments?",
            min_value=1,
            max_value=5,
            value=3
        )
        submit = st.form_submit_button("Submit survey")

    if submit:
        code = survey_code()
        header = [
            "sid", "pid", "cond", "sync_ai", "n_trials",
            "study_minutes", "confidence_1to5", "code", "ts"
        ]
        study_minutes = round((time.time() - st.session_state.start_ts) / 60.0, 2)
        row_log = {
            "sid": st.session_state.sid,
            "pid": st.session_state.pid,
            "cond": st.session_state.cond,
            "sync_ai": int(bool(st.session_state.sync_ai)),
            "n_trials": n_trials,
            "study_minutes": study_minutes,
            "confidence_1to5": conf,
            "code": code,
            "ts": datetime.utcnow().isoformat()
        }
        write_csv(LOG_SESS, row_log, header)
        st.session_state.survey_done = True
        st.session_state.code = code
        st.session_state.page = "finish"
        st.rerun()


def page_finish():
    st.title(APP_TITLE)
    st.subheader("Thank you for participating!")

    if not st.session_state.survey_done:
        st.warning("You have not completed the study yet.")
        if st.button("Back to start"):
            st.session_state.page = "home"
            st.rerun()
        return

    st.write("Your completion code is:")
    st.code(st.session_state.code or "VS-A3-XXXXXX")

    if st.button("Start over"):
        # reset everything
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_state()
        st.session_state.page = "home"
        st.rerun()


# ---------- Main ----------

def main():
    init_state()
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "consent":
        page_consent()
    elif st.session_state.page == "task":
        page_task()
    elif st.session_state.page == "survey":
        page_survey()
    elif st.session_state.page == "finish":
        page_finish()
    else:
        st.session_state.page = "home"
        page_home()


if __name__ == "__main__":
    main()

