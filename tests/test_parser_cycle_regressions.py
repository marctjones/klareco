from klareco.parser import parse


def test_article_licensed_adjective_pp_does_not_cycle_through_coordination():
    text = (
        "Flanke al tio, konsiderendas la malsimileco inter la kvanto de datumoj "
        "por trejni la AI-motoron je dispono en la angla kaj la datokvanto en "
        "ĉiuj aliaj lingvoj, kaj, aparte grave, en malgrandaj lingvoj kiel la "
        "bosna kaj Esperanto – ”malgrandaj” laŭ la vidpunkto de la ĉeesto en la "
        "bitmondo, kiel precizigis prave profesoro en la Universitato de Sarajevo "
        "pri la bosna, serba kaj kroata lingvoj, Halid Bulić."
    )
    ast = parse(text)
    tokens = ast["vortoj"]
    bosna = next(word for word in tokens if word["plena_vorto"] == "bosna")
    esperanto = next(word for word in tokens if word["plena_vorto"] == "Esperanto")
    kiel = next(word for word in tokens if word["plena_vorto"] == "kiel")
    assert kiel["rolo"] == "case"
    assert kiel["kapo"] == bosna["id"]
    assert esperanto["rolo"] == "conj"
    assert esperanto["kapo"] == bosna["id"]
