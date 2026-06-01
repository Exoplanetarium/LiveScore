"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ThemedText = ThemedText;
const react_native_1 = require("react-native");
const useThemeColor_1 = require("../hooks/useThemeColor");
function ThemedText({ style, lightColor, darkColor, type = 'default', ...rest }) {
    const color = (0, useThemeColor_1.useThemeColor)({ light: lightColor, dark: darkColor }, 'text');
    return (React.createElement(react_native_1.Text, { style: [
            { color },
            type === 'default' ? styles.default : undefined,
            type === 'title' ? styles.title : undefined,
            type === 'defaultSemiBold' ? styles.defaultSemiBold : undefined,
            type === 'subtitle' ? styles.subtitle : undefined,
            type === 'link' ? styles.link : undefined,
            style,
        ], ...rest }));
}
const styles = react_native_1.StyleSheet.create({
    default: {
        fontSize: 16,
        lineHeight: 24,
    },
    defaultSemiBold: {
        fontSize: 16,
        lineHeight: 24,
        fontWeight: '600',
    },
    title: {
        fontSize: 32,
        fontWeight: 'bold',
        lineHeight: 32,
    },
    subtitle: {
        fontSize: 20,
        fontWeight: 'bold',
    },
    link: {
        lineHeight: 30,
        fontSize: 16,
        color: '#0a7ea4',
    },
});
