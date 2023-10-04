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

    def print(self):
        print("Tag:", self.tag, " Attrs:", self.attrs)


def get_html_attribute(tag, name):
    parser = HTMLTagParser()
    parser.feed(tag)

    if type(name) == list:
        value = []
        for s in name:
            v = parser.get_attr(s)
            value.append(v)
    else:
        value = parser.get_attr(name)
    return value

def split_html_content(content, tag, index=0):
    if content is None:
        return None, None, None

    start = content.find("<%s" % tag, index)
    if start < 0:
        return None, None, content
    end = content.find("</%s>" % tag, start)
    if end >= 0:
        end += len("</%s>" % tag)
    else:
        end1 = content.find("/>", start)
        end2 = content.find(">", start)
        if end1 < end2:
            end = end1 + 2
        else:
            end = end2 + 1
    return content[:start], content[start:end], content[end:]


def get_inner_html(tag, tagName):
    tag_start = tag.find("<%s" % tagName)
    inner_start = tag.find(">", tag_start) + 1
    tag_end = tag.find("</%s>" % tagName)
    return tag[inner_start:tag_end]


def process_include_html(content, params={}, template_content={}):
    if content is None or content == "":
        return ""

    content = content.replace("<!LP-", "<LP-")
    content = content.replace("</!LP-", "</LP-")

    p = content.find("<LP-")
    if p < 0:
        return content

    if content[p+1:].startswith("LP-template"):
        p1, tag, p2 = split_html_content(content, "LP-template", p)
        use_template = get_html_attribute(tag, "use")
        template_filepath = "ui/template/%s.html" % use_template

        p1, tag, p2 = split_html_content(p2, "LP-param")
        while tag is not None:
            values = get_html_attribute(tag, ["name", "value"])
            params[values[0]] = values[1]
            p1, tag, p2 = split_html_content(p2, "LP-param")

        p1, tag, p2 = split_html_content(p2, "LP-content")
        while tag is not None:
            content_name = get_html_attribute(tag, "name")
            template_content[content_name] = get_inner_html(tag, "LP-content")
            p1, tag, p2 = split_html_content(p2, "LP-content")

        with open(template_filepath, "rt", encoding="utf-8") as fp:
            base_content = fp.read()
            return process_include_html(base_content, params, template_content)
    elif content[p+1:].startswith("LP-include-content"):
        p1, tag, p2 = split_html_content(content, "LP-include-content", p)
        content_name = get_html_attribute(tag, "name")
        if content_name in template_content:
            sub_content = template_content[content_name]
        else:
            default_content = get_inner_html(tag, "LP-include-content")
            sub_content = default_content
        sub_content = process_include_html(sub_content, params, template_content)
        p2 = process_include_html(p2, params, template_content)
    elif content[p+1:].startswith("LP-include-html"):
        p1, tag, p2 = split_html_content(content, "LP-include-html", p)
        src_name = get_html_attribute(tag, "src")
        src_file_path = "ui/fragment/%s" % src_name
        if not os.path.exists(src_file_path):
            src_file_path = "ui/dialog/%s" % src_name
        if os.path.exists(src_file_path):
            with open(src_file_path, "rt", encoding="UTF-8") as fp:
                sub_content = fp.read()
            sub_content = process_include_html(sub_content, params, template_content)
        p2 = process_include_html(p2, params, template_content)
    elif content[p + 1:].startswith("LP-include-string"):
        p1, tag, p2 = split_html_content(content, "LP-include-string", p)
        var_name = get_html_attribute(tag, "var")
        if var_name in params:
            sub_content = params[var_name]
            sub_content = process_include_html(sub_content, params, template_content)
        else:
            print("var name not found:", var_name)
            sub_content = ''
        p2 = process_include_html(p2, params, template_content)
    else:
        print("Unknown LP Tag:", content[p:p + 32])
        p1 = content[:p]
        sub_content = ' Content Parse Error: '
        p2 = content[p:p + 32]
    content = p1 + sub_content + p2
    return content
