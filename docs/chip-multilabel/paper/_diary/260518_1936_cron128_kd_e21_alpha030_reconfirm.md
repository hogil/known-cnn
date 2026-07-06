# 260518 19:36 — cron #128 KD_E21 α=0.30 reconfirm

Appended one-line update to §5.49.5 in `05_experiments.md` reconfirming
KD_E21 α=0.30 single-teacher KD result at cron #128 measurement window:
POS9 bit_F1 I10 = 0.8886 / 0.08 % Total FAR, I13 = 0.8096 / 0.00 % Total
FAR — intra-KD-batch best, still −0.1067 bit_F1 below §5.49.4 4-way
champion (E7+LS20_s77 0.9953 / 0.00 %). Champion table unchanged.
KD-axis search remains closed.

WHY paper-worth: closes the second tick on the KD ceiling, confirming
the cron #122 reading was not a one-shot artefact and bounding the
KD-as-standalone path on both ends (ensemble-teacher failure §5.49.cron
#79; single-teacher ceiling cron #122/#128).
