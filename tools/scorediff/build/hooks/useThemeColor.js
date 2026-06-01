"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.useThemeColor = useThemeColor;
const react_native_1 = require("react-native");
const Colors_1 = require("../constants/Colors");
function useThemeColor(props, colorName) {
    var _a;
    const theme = (_a = (0, react_native_1.useColorScheme)()) !== null && _a !== void 0 ? _a : 'light';
    const colorFromProps = props[theme];
    if (colorFromProps) {
        return colorFromProps;
    }
    else {
        return Colors_1.Colors[theme][colorName];
    }
}
