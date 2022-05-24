import json

textkeys = [
    "cclom:title",
    "cm:title",
    "cm:name",
    "cclom:general_description",
    "cm:description",
]
# kwkeys = ["cclom:general_keyword"]
urlkeys = ["ccm:wwwurl"]

csv = open("wirlernenonline2-dedup.txt", "w")


def getText(props):
    text = ""
    for k in textkeys:
        if k in props.keys():
            val = props[k]
            if isinstance(val, list):
                val = " ".join(val)
            text = text + " " + val
    # if kwkeys[0] in props.keys():
    #    text = text + " " + " ".join(props[kwkeys[0]])
    return text.replace('"', "").strip()


def getUrl(props):
    url = "_"
    for k in urlkeys:
        if k in props.keys():
            url = props[k]
    return url.replace('"', "").strip().replace(" ", "+")


def valid(json_data):
    if "ccm:collection_io_reference" in json_data.get("_source", None).get("aspects"):
        return False
    # filter collections
    if json_data.get("_source", None).get("type") != "ccm:io":
        return False

    # filter archived and other data
    if (
        json_data.get("_source", None).get("nodeRef").get("storeRef").get("protocol")
        != "workspace"
    ):
        return False

    if (
        json_data.get("_source", None).get("properties").get("cclom:format")
        == "application/zip"
    ):
        return False

    if json_data.get("_source", None).get("properties").get("cclom:title") is None:
        return False

    if json_data.get("_source", None).get("owner") == "WLO-Upload":
        return False

    if (
        json_data.get("_source", None).get("properties").get("cm:edu_metadataset")
        != "mds_oeh"
    ):
        return False

    return True


with open("wirlernenonline_20_05_2021.json") as f:
    for line in f:
        jline = json.loads(line)
        id = jline["_source"]["nodeRef"]["id"]
        props = jline["_source"]["properties"]
        if valid(jline):
            text = getText(props)
            url = getUrl(props)
            if text.strip() != id:
                csv.write(
                    id
                    + " "
                    + url
                    + " "
                    + text.replace("\n", " ").replace("\r", "")
                    + "\n"
                )
csv.close()
