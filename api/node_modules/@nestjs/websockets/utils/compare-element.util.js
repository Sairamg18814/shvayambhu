"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.compareElementAt = compareElementAt;
function compareElementAt(prev, curr, index) {
    return prev && curr && prev[index] === curr[index];
}
