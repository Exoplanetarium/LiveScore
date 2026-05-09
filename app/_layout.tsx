import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";

export default function RootLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: "#2f95dc",
        headerShown: false,
      }}
      initialRouteName="index"
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Live",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "radio" : "radio-outline"}
              color={color}
              size={28}
            />
          ),
        }}
      />
      <Tabs.Screen
        name="classic"
        options={{
          title: "Classic",
          tabBarIcon: ({ color, focused }) => (
            <Ionicons
              name={focused ? "disc" : "disc-outline"}
              color={color}
              size={28}
            />
          ),
        }}
      />
      <Tabs.Screen name="index_old" options={{ href: null }} />
      <Tabs.Screen name="index_backup" options={{ href: null }} />
    </Tabs>
  );
}
