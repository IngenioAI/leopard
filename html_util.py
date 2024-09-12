import os
from html.parser import HTMLParser


class HTMLElement():
    def __init__(self, tag_name) -> None:
        self.tag_name = tag_name
        self.attrs = {}
        self.parent = None
        self.data = ""
        self.children = []
        self.start_pos = None
        self.end_pos = None
        self.inner_html = ""

    def add_child(self, elem):
        elem.parent = self
        self.children.append(elem)

    def print(self, recur=False, indent=0):
        print(" " * indent, "Tag:", self.tag_name, " Attrs:", self.attrs, " Data:", self.data)
        if recur:
            for child in self.children:
                child.print(recur, indent + 1)


class HTMLTagParser(HTMLParser):
    def __init__(self, data="", mode="simple"):
        super().__init__(convert_charrefs=True)
        self.roots = []
        self.current = None
        self.data = data
        self.mode = mode  # simple, normal
        super().feed(data)

    def feed(self, data):
        self.data += data
        super().feed(data)

    def handle_starttag(self, tag, attrs):
        elem = HTMLElement(tag)
        for (key, value) in attrs:
            elem.attrs[key] = value
        if len(self.roots) == 0:
            self.roots.append(elem)

        if self.mode == "normal":
            elem.start_pos = (self.getpos()[0], self.getpos()[1] + len(self.get_starttag_text()))
            if self.current is None:
                self.roots.append(elem)
            else:
                self.current.add_child(elem)
            self.current = elem

    def handle_endtag(self, tag):
        if self.current is not None:
            while tag != self.current.tag_name:
                print("End tag mismatch:", tag, self.current.tag_name)
                self.current = self.current.parent
            self.current.end_pos = self.getpos()
            self.current.inner_html = ""
            lines = self.data.splitlines(True)
            start = self.current.start_pos
            end = self.current.end_pos
            for r in range(start[0], end[0] + 1):
                if r == start[0] and r == end[0]:
                    self.current.inner_html += lines[r - 1][start[1]:end[1]]
                elif r == start[0]:
                    self.current.inner_html += lines[r - 1][start[1]:]
                elif r == end[0]:
                    self.current.inner_html += lines[r - 1][:end[1]]
                else:
                    self.current.inner_html += lines[r - 1]
            self.current = self.current.parent

    def handle_data(self, data):
        if self.current is not None:
            self.current.data = data

    def get_attr(self, name):
        if name in self.roots[0].attrs:
            return self.roots[0].attrs[name]
        return ""

    def print(self, recur=False):
        for root in self.roots:
            root.print(recur, 0)

    def error(self, message):
        print("ParserError:", message)


def get_html_attribute(tag, name):
    parser = HTMLTagParser(tag)
    # parser.print()

    if isinstance(name, list):
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

    tag_open = f'<{tag}'
    tag_close = f'</{tag}>'

    start = content.find(tag_open, index)
    if start < 0:
        return None, None, content
    starttag = content.find(">", start + 1)
    startendtag = content.find("/>", start + 1)
    if 0 <= startendtag < starttag:
        end = startendtag + 2
    else:
        starttag += 1
        end = content.find(tag_close, starttag)
        if end > 0:
            open_count = content.count(tag_open, start + 1, end)
            while open_count > 0:
                end = content.find(tag_close, end + 1)
                open_count -= 1
            end += len(tag_close)
        else:
            end = len(content)
            # print("tag splitted by eof:", tag, content[start:end])

    return content[:start], content[start:end], content[end:]


def get_inner_html(tag, tag_name):
    tag_open = f'<{tag_name}'
    tag_close = f'</{tag_name}>'
    open_count = tag.count(tag_open)
    tag_start = tag.find(tag_open)
    inner_start = tag.find(">", tag_start) + 1
    tag_end = tag.find(tag_close, inner_start)
    if tag_end >= 0:
        open_count = tag.count(tag_open, inner_start)
        while open_count > 0:
            tag_end = tag.find(tag_close, tag_end + 1)
            open_count -= 1
        return tag[inner_start:tag_end]
    print("No inner HTML for:", tag)
    return ""


def process_template(tag, template_content, params):
    new_params = dict.copy(params)
    new_template_content = dict.copy(template_content)

    use_template = get_html_attribute(tag, "use")
    # print("use template:", use_template)
    template_filepath = f'ui/template/{use_template}.html'

    inner_html = get_inner_html(tag, "LP-template")

    _, tag, p2 = split_html_content(inner_html, "LP-param")
    while tag is not None:
        values = get_html_attribute(tag, ["name", "value"])
        new_params[values[0]] = values[1]
        _, tag, p2 = split_html_content(p2, "LP-param")

    _, tag, p2 = split_html_content(inner_html, "LP-content")
    while tag is not None:
        content_name = get_html_attribute(tag, "name")
        # print("add template content:", content_name)
        new_template_content[content_name] = get_inner_html(tag, "LP-content")
        _, tag, p2 = split_html_content(p2, "LP-content")

    with open(template_filepath, "rt", encoding="utf-8") as fp:
        base_content = fp.read()

    return process_lp_html(base_content, new_params, new_template_content)


def process_include_content(tag, template_content):
    content_name = get_html_attribute(tag, "name")
    if content_name in template_content:
        # print("get template content:", content_name)
        sub_content = template_content[content_name]
    else:
        # print("get template content (default):", content_name)
        default_content = get_inner_html(tag, "LP-include-content")
        sub_content = default_content
    return sub_content


def process_include_html_tag(tag):
    src_name = get_html_attribute(tag, "src")
    # print("Load template:", src_name)
    src_file_path = f'ui/fragment/{src_name}'
    if not os.path.exists(src_file_path):
        src_file_path = f'ui/dialog/{src_name}'
    if os.path.exists(src_file_path):
        with open(src_file_path, "rt", encoding="UTF-8") as fp:
            return fp.read()
    return ""


def process_include_script(tag, template_content):
    src_name = get_html_attribute(tag, "src")
    if src_name == "":
        file_title, _ = os.path.splitext(template_content["file_path"])
        src_name = f'/js/page/{file_title}.js'
    script_path = os.path.join("webroot", src_name[1:] if src_name[0] == "/" else src_name)
    if os.path.exists(script_path):
        return f'<script type="module" src="{src_name}" charset="utf-8"></script>'
    return ""


def process_include_dialog_script(tag, params):
    src_name = get_html_attribute(tag, "src")
    if src_name == "":
        file_title = params['id']
        src_name = f'/js/dialog/{file_title}.js'
    script_path = os.path.join("webroot", src_name[1:] if src_name[0] == "/" else src_name)
    if os.path.exists(script_path):
        return f'<script type="module" src="{src_name}" charset="utf-8"></script>'
    return ""


def process_include_string(tag, template_content, params):
    var_name = get_html_attribute(tag, "var")
    if var_name in params:
        sub_content = params[var_name]
        sub_content = process_lp_html(sub_content, params, template_content)
    else:
        default_var = get_html_attribute(tag, "default")
        if default_var is not None:
            sub_content = default_var
        else:
            print("var name not found (no default):", var_name)
            sub_content = ''
    return sub_content


def process_lp_html(content, params=None, template_content=None):
    if content is None or content == "":
        return ""

    if params is None:
        params = {}

    if template_content is None:
        template_content = {}

    content = content.replace("<!LP-", "<LP-")
    content = content.replace("</!LP-", "</LP-")

    p = content.find("<LP-")
    if p < 0:
        return content

    if content[p + 1:].startswith("LP-template"):
        p1, tag, p2 = split_html_content(content, "LP-template", p)
        sub_content = process_template(tag, template_content, params)
    elif content[p + 1:].startswith("LP-include-content"):
        p1, tag, p2 = split_html_content(content, "LP-include-content", p)
        # print("include-content", tag)
        sub_content = process_include_content(tag, template_content)
        sub_content = process_lp_html(sub_content, params, template_content)
    elif content[p + 1:].startswith("LP-include-html"):
        p1, tag, p2 = split_html_content(content, "LP-include-html", p)
        sub_content = process_include_html_tag(tag)
        sub_content = process_lp_html(sub_content, params, template_content)
    elif content[p + 1:].startswith("LP-include-script"):
        p1, tag, p2 = split_html_content(content, "LP-include-script", p)
        sub_content = process_include_script(tag, template_content)
    elif content[p + 1:].startswith("LP-include-dialog-script"):
        p1, tag, p2 = split_html_content(content, "LP-include-dialog-script", p)
        sub_content = process_include_dialog_script(tag, params)
    elif content[p + 1:].startswith("LP-include-string"):
        p1, tag, p2 = split_html_content(content, "LP-include-string", p)
        sub_content = process_include_string(tag, template_content, params)
    else:
        print("Unknown LP Tag:", content[p:p + 32])
        p1 = content[:p]
        sub_content = ' Content Parse Error: '
        p2 = content[p:p + 32]
    p2 = process_lp_html(p2, params, template_content)
    content = p1 + sub_content + p2
    return content
