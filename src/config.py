def get_config(dataset):
    config_dict = {}
    config_dict["seed"] = 18
    config_dict["bootstrap_size"] = 1000
    config_dict["boot_iter"] = 1000
    config_dict["pval_threshold"] = 0.05
    config_dict["max_bins"] = 5

    if dataset == "kidney":
        config_dict["target"] = "class"
        config_dict["test_size"] = 1 / 3
        config_dict["corr_threshold"] = 0.7
        config_dict["feature_clf"] = {
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
        config_dict["label_dict"] = {
            "normal": 0,
            "abnormal": 1,
            "notpresent": 0,
            "present": 1,
            "yes": 1,
            "no": 0,
            "ckd": 1,
            "notckd": 0,
            "good": 0,
            "poor": 1,
        }
    elif dataset == "adult":
        config_dict["target"] = "income"
        config_dict["test_size"] = 31 / 32
        config_dict["corr_threshold"] = 0.6
        config_dict["ft_mod_dict"] = {
            # "race": [" White", " Black", " Asian-Pac-Islander", " Amer-Indian-Eskimo", " Other"],
        }
        config_dict["feature_clf"] = {
            "occupation_cat": "cat",
            "marital_status_cat": "cat",
            "workclass_gov": "cat",
            "workclass_private": "cat",
            "workclass_cat": "cat",
            "education_cat": "cat",
            "age": "num",
            "fnlwgt": "num",
            "occupation": "class",
            "workclass": "class",
            "marital_status": "class",
            "relationship": "class",
            "race": "cat",
            "sex": "cat",
            "capital_gain": "num",
            "capital_loss": "num",
            "hours_per_week": "num",
            "native_country": "class",
            "income": "cat",
        }
        config_dict["label_dict"] = {
            " >50K": 0,
            " <=50K": 1,
            " Male": 1,
            " Female": 0,
        }
        config_dict["workclass_gov"] = {
            " State-gov": 3,
            " Federal-gov": 2,
            " Local-gov": 1,
            " Self-emp-inc": 0,
            " Self-emp-not-inc": 0,
            " Private": 0,
            " Without-pay": 0,
            " Never-worked": 0,
            " ?": 0,
        }
        config_dict["workclass_private"] = {
            " State-gov": 0,
            " Federal-gov": 0,
            " Local-gov": 0,
            " Self-emp-inc": 3,
            " Self-emp-not-inc": 2,
            " Private": 1,
            " Without-pay": 0,
            " Never-worked": 0,
            " ?": 0,
        }
        config_dict["marital_status_cat"] = {
            " Never-married": 0,
            " Married-civ-spouse": 1,
            " Divorced": 0,
            " Married-spouse-absent": 0,
            " Separated": 0,
            " Married-AF-spouse": 1,
            " Widowed": 0,
        }
        config_dict["occupation_cat"] = {
            " Adm-clerical": 0,
            " Exec-managerial": 3,
            " Handlers-cleaners": 0,
            " Prof-specialty": 3,
            " Other-service": 0,
            " Sales": 1,
            " Craft-repair": 1,
            " Transport-moving": 1,
            " Farming-fishing": 0,
            " Machine-op-inspct": 0,
            " Tech-support": 2,
            " ?": 0,
            " Protective-serv": 2,
            " Armed-Forces": 0,
            " Priv-house-serv": 0,
        }
    elif dataset == "diagnosis":
        config_dict["target"] = "inflammation"
        config_dict["test_size"] = 1 / 3
        config_dict["corr_threshold"] = 0.7
        config_dict["feature_clf"] = {
            "temp": "num",
            "nausea": "cat",
            "lumbar_pain": "cat",
            "urine_pushing": "cat",
            "micturition_pain": "cat",
            "burning_urethra": "cat",
            "inflammation": "cat",
            "nephritis": "cat",
        }
        config_dict["label_dict"] = {
            "no": 0,
            "yes": 1,
        }
    elif dataset == "breast_cancer":
        config_dict["target"] = "class"
        config_dict["test_size"] = 1 / 3
        config_dict["corr_threshold"] = 0.6
        config_dict["feature_clf"] = {
            "class": "cat",
            "age": "cat",
            "menopause": "cat",
            "tumor_size": "cat",
            "inv_nodes": "cat",
            "node_caps": "cat",
            "deg_malig": "cat",
            "breast": "class",
            "breast_quad": "class",
            "irradiat": "cat",
        }
        config_dict["label_dict"] = {
            "no-recurrence-events": 0,
            "recurrence-events": 1,
            "yes": 1,
            "no": 0,
            "premeno": 0,
            "lt40": 1,
            "ge40": 1,
        }
    return config_dict
