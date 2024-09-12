import os
import json
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def clear_progress():
    if os.path.exists("/apprun/progress.json"):
        os.remove("/apprun/progress.json")


def save_progress(obj):
    with open("/apprun/progress.json", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)


def save_result(obj, filename):
    with open(f"/data/output/{filename}", "wt", encoding="UTF-8") as fp:
        json.dump(obj, fp, indent=4)
