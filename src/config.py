seed = 18
test_size = 1 / 3

bootstrap_size = 1000
boot_iter = 1000
pval_threshold = 0.05
max_bins = 5


feature_clf = {
    "age": "num",
    "bp": "num",
    "sg": "cat",
    "al": "cat",
    "su": "cat",
    "rbc": "cat",
    "pc": "cat",
    "pcc": "cat",
    "ba": "cat",
    "bgr": "num",
    "bu": "num",
    "sc": "num",
    "sod": "num",
    "pot": "num",
    "hemo": "num",
    "pcv": "num",
    "wbcc": "num",
    "rbcc": "num",
    "htn": "cat",
    "dm": "cat",
    "cad": "cat",
    "appet": "cat",
    "pe": "cat",
    "ane": "cat",
    "class": "cat",
}
label_dict = {"normal": 0, "abnormal": 1, "notpresent": 0, "present": 1, "yes": 1, "no": 0, "ckd": 1, "notckd": 0, "good": 0, "poor": 1}
