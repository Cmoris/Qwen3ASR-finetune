# ── Special tokens ────────────────────────────────────────────────────────────
TS_TOKEN = "<ts>"
TE_TOKEN = "<te>"
BC_TOKEN = "<bc>"
PAUSE_TOKEN = "<pause>"
SILENCE_TOKEN = "<silence>"

SPEAKER_TOKENS = {
    "A": ["<speaker_A>", "</speaker_A>"],
    "B": ["<speaker_B>", "</speaker_B>"],
}
STREAMING_CONT = " ..."   # suffix meaning "transcript not yet complete"

# ── Default hyper-params (mirror DataArguments style) ────────────────────────
DEFAULT_CHUNK_SECS   = 0.5    # seconds of audio per streaming step
DEFAULT_SAMPLE_RATE  = 16_000

DEFAULT_CONTEXT_LENGTH = 1
