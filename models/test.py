# fastscore.input: gbm_input
# fastscore.output: gbm_output

def action(datum):
    score = list(gbmFit.predict(pd.DataFrame([datum])))[0]
    yield score