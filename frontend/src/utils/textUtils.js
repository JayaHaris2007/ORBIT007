// The local model occasionally emits malformed keycap numbers — a digit followed
// by variation selectors, combining marks, dingbats or braille patterns — which
// render as stray glyphs (1️⃣, 2️⃝, 4️⠁, 5️⣇). Normalize before markdown rendering.
const BROKEN_LIST_MARKER = /^[ \t]*(\d{1,2})[\uFE0F\u20D0-\u20FF\u2460-\u24FF\u2700-\u27BF\u2800-\u28FF]+[ \t]*(?=\S)/gm;
const INVISIBLE_MARKS = /[\uFE00-\uFE0F\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFFFD]/g;
const NUMBERED_ITEM = /^[ \t]*\d{1,2}\.\s/;

// Keep a numbered section from being glued onto the paragraph above it.
const spaceNumberedItems = (text) => {
    const lines = [];
    text.split('\n').forEach((line) => {
        const previous = lines[lines.length - 1];
        if (NUMBERED_ITEM.test(line) && previous?.trim() && !NUMBERED_ITEM.test(previous)) {
            lines.push('');
        }
        lines.push(line);
    });
    return lines.join('\n');
};

export const cleanAssistantText = (value) => {
    if (typeof value !== 'string' || !value) return '';
    return spaceNumberedItems(
        value
            .replace(BROKEN_LIST_MARKER, '$1. ')
            .replace(INVISIBLE_MARKS, '')
            .replace(/\n{3,}/g, '\n\n')
    ).trim();
};
