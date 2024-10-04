export class CSV {
    static parse(text, quote='|', returnCellInfo=false) {
        let p = '', row = [''], ret = [row], i = 0, r = 0, s = !0, l;
        let c = 0, cx = 0, cy = 0, cells = [[0, null]], lines =[];
        for (l of text) {
            if (quote === l) {
                if (s && l === p)
                    row[i] += l;
                s = !s;
            }
            else if (',' === l && s) {
                l = row[++i] = '';
                cells[cx][1] = c;
                cx += 1;
                cells[cx] = [c+1, null];
            }
            else if ('\n' === l && s) {
                if ('\r' === p)
                    row[i] = row[i].slice(0, -1);
                row = ret[++r] = [l = '']; i = 0;

                cells[cx][1] = c;
                lines[cy] = cells;
                cy += 1;
                cells = [[c+1, null]];
                cx = 0;
            }
            else
                row[i] += l;
            p = l;
            c++;
        }
        if (returnCellInfo) {
            return {
                data: ret,
                cellInfo: lines
            }
        }
        return ret;
    }
}
