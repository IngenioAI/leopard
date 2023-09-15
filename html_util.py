import os
from html.parser import HTMLParser


class HTMLTagParser(HTMLParser):
    def __init__(self, *, convert_charrefs: bool = True) -> None:
        super().__init__(convert_charrefs=convert_charrefs)
        self.tag = ''
        self.attrs = dict()

    def handle_starttag(self, tag, attrs):
        self.tag = tag
        for (key, value) in attrs:
            self.attrs[key] = value

    def get_attr(self, name):
        return self.attrs[name]


def split_html_content(content, tag):
    index = 0
    start = content.find("<%s" % tag, index)
    if start < 0:
        return content, None, None
    end = content.find("</%s>" % tag, start) + len("</%s>" % tag)
    if end < 0:
        return content[:start], content[start:], None
    return content[:start], content[start:end], content[end:]


def process_include_html(content, params=None):
    p = content.find("<LP-")
    if p < 0:
        return content

    if content[p + 1:].startswith("LP-include-html"):
        p1, tag, p2 = split_html_content(content, "LP-include-html")
        parser = HTMLTagParser()
        parser.feed(tag)
        src_file_path = "ui/fragment/%s" % parser.get_attr('src')
        if not os.path.exists(src_file_path):
            src_file_path = "ui/dialog/%s" % parser.get_attr('src')
        if os.path.exists(src_file_path):
            with open(src_file_path, "rt", encoding="UTF-8") as fp:
                sub_content = fp.read()
            sub_content = process_include_html(sub_content, params)
        if p2 is not None:
            p2 = process_include_html(p2, params)
        else:
            p2 = ''
    elif content[p + 1:].startswith("LP-include-string"):
        p1, tag, p2 = split_html_content(content, "LP-include-string")
        parser = HTMLTagParser()
        parser.feed(tag)
        var_name = parser.get_attr('var')
        if params is not None:
            sub_content = params[var_name]
            sub_content = process_include_html(sub_content, params)
        else:
            sub_content = ''
        if p2 is not None:
            p2 = process_include_html(p2, params)
        else:
            p2 = ''
    else:
        print("Unknown LP Tag:", content[p:p + 32])
        p1 = content[:p]
        sub_content = ' Content Parse Error: '
        p2 = content[p:p + 32]
    content = p1 + sub_content + p2
    return content
