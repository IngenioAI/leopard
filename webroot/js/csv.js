export class CSV {
    static parse(text, quote='|') {
        let p = '', row = [''], ret = [row], i = 0, r = 0, s = !0, l;
        for (l of text) {
            if (quote === l) {
                if (s && l === p)
                    row[i] += l;
                s = !s;
            }
            else if (',' === l && s)
                l = row[++i] = '';
            else if ('\n' === l && s) {
                if ('\r' === p)
                    row[i] = row[i].slice(0, -1);
                row = ret[++r] = [l = '']; i = 0;
            }
            else
                row[i] += l;
            p = l;
        }
        return ret;
    }
}
