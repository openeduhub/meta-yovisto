import json
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path):
    print("File '" + path + "' does not exits.")
    sys.exit(1)

textkeys = [
    "cclom:title",
    "cm:title",
    "cm:name",
    "cclom:general_description",
    "cm:description",
]
kwkeys = ["cclom:general_keyword"]
csv = open(path.replace(".json", ".csv"), "w")


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


def getText(props):
    text = ""
    for k in textkeys:
        if k in props.keys():
            val = props[k]
            if isinstance(val, list):
                val = " ".join(val)
            text = text + " " + val
    if kwkeys[0] in props.keys():
        text = text + " " + " ".join(props[kwkeys[0]])
    return text.replace('"', "")


with open(path) as f:
    for line in f:
        jline = json.loads(line)
        if valid(jline):
            id = jline["_source"]["nodeRef"]["id"]
            props = jline["_source"]["properties"]
            text = getText(props)
            csv.write('"' + text.replace("\n", " ") + '","' + id + '"\n')

csv.close()
