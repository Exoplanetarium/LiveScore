import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useFocusEffect } from "expo-router";
import React, { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  View,
} from "react-native";
import PianoSheetMusic from "../components/PianoSheetMusic";
import { ThemedText } from "../components/ThemedText";
import {
  exportScoreAsMidi,
  exportScoreAsMusicXml,
} from "../lib/scoreExport";
import {
  deleteSavedScore,
  listSavedScores,
  loadSavedScore,
  type SavedScore,
  type SavedScoreMeta,
} from "../lib/savedScores";

function formatTimestamp(ms: number) {
  try {
    return new Date(ms).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return new Date(ms).toISOString();
  }
}

function formatLength(seconds: number) {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

export default function LibraryScreen() {
  const [scores, setScores] = useState<SavedScoreMeta[]>([]);
  const [isLoadingList, setIsLoadingList] = useState(true);
  const [selected, setSelected] = useState<SavedScore | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const refreshList = useCallback(async () => {
    setIsLoadingList(true);
    try {
      setScores(await listSavedScores());
    } finally {
      setIsLoadingList(false);
    }
  }, []);

  // Reload whenever the tab regains focus so a recording just saved from the
  // Live tab shows up without needing a manual refresh.
  useFocusEffect(
    useCallback(() => {
      refreshList();
    }, [refreshList]),
  );

  const openScore = useCallback(async (id: string) => {
    setIsLoadingDetail(true);
    try {
      const record = await loadSavedScore(id);
      if (record) {
        setSelected(record);
      } else {
        Alert.alert("Unavailable", "This recording could not be opened.");
      }
    } finally {
      setIsLoadingDetail(false);
    }
  }, []);

  const confirmDelete = useCallback(
    (meta: SavedScoreMeta) => {
      Alert.alert(
        "Delete recording",
        `Delete “${meta.title}”? This cannot be undone.`,
        [
          { text: "Cancel", style: "cancel" },
          {
            text: "Delete",
            style: "destructive",
            onPress: async () => {
              await deleteSavedScore(meta.id);
              if (selected?.id === meta.id) {
                setSelected(null);
              }
              refreshList();
            },
          },
        ],
      );
    },
    [refreshList, selected],
  );

  const runExport = useCallback(
    async (kind: "midi" | "musicxml") => {
      if (!selected) {
        return;
      }
      setIsExporting(true);
      try {
        if (kind === "midi") {
          await exportScoreAsMidi(selected.analysis, selected.bpm, selected.title);
        } else {
          await exportScoreAsMusicXml(
            selected.analysis,
            selected.bpm,
            selected.title,
          );
        }
      } catch (error: any) {
        Alert.alert("Export Failed", error?.message || "Could not export.");
      } finally {
        setIsExporting(false);
      }
    },
    [selected],
  );

  if (selected) {
    return (
      <LinearGradient
        colors={["#04070f", "#0b1220", "#111c30"]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.screen}
      >
        <View style={styles.detailTopBar}>
          <TouchableOpacity
            style={styles.backButton}
            onPress={() => setSelected(null)}
            hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
          >
            <Ionicons name="chevron-back" size={20} color="#f8fafc" />
            <ThemedText style={styles.backButtonText}>Library</ThemedText>
          </TouchableOpacity>
          <View style={styles.detailTitleWrap}>
            <ThemedText
              style={styles.detailTitle}
              lightColor="#f8fafc"
              darkColor="#f8fafc"
              numberOfLines={1}
            >
              {selected.title}
            </ThemedText>
            <ThemedText style={styles.detailSubtitle}>
              {Math.round(selected.bpm)} bpm ·{" "}
              {selected.noteCount + selected.chordCount} events ·{" "}
              {formatLength(selected.durationSeconds)}
            </ThemedText>
          </View>
        </View>

        <View style={styles.detailScore}>
          <PianoSheetMusic results={selected.analysis} liveFollow={false} />
        </View>

        <LinearGradient
          colors={["rgba(15,23,42,0.96)", "rgba(30,41,59,0.92)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.exportCard}
        >
          <ThemedText
            style={styles.exportCardTitle}
            lightColor="#f8fafc"
            darkColor="#f8fafc"
          >
            Export
          </ThemedText>
          <View style={styles.exportRow}>
            <TouchableOpacity
              style={[styles.exportActionButton, styles.exportActionButtonFlex]}
              onPress={() => runExport("midi")}
              disabled={isExporting}
            >
              <Ionicons name="musical-notes-outline" size={18} color="#ffffff" />
              <ThemedText style={styles.exportActionButtonText}>MIDI</ThemedText>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.exportActionButton, styles.exportActionButtonFlex]}
              onPress={() => runExport("musicxml")}
              disabled={isExporting}
            >
              <Ionicons name="document-text-outline" size={18} color="#ffffff" />
              <ThemedText style={styles.exportActionButtonText}>
                MusicXML
              </ThemedText>
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            style={styles.deleteButton}
            onPress={() => confirmDelete(selected)}
            disabled={isExporting}
          >
            <Ionicons name="trash-outline" size={16} color="#fca5a5" />
            <ThemedText style={styles.deleteButtonText}>
              Delete recording
            </ThemedText>
          </TouchableOpacity>
        </LinearGradient>
      </LinearGradient>
    );
  }

  return (
    <LinearGradient
      colors={["#04070f", "#0b1220", "#111c30"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
      style={styles.screen}
    >
      <View style={styles.header}>
        <ThemedText
          style={styles.headerTitle}
          lightColor="#f8fafc"
          darkColor="#f8fafc"
        >
          Library
        </ThemedText>
        <ThemedText style={styles.headerSubtitle}>
          Saved recordings
        </ThemedText>
      </View>

      {isLoadingList && scores.length === 0 ? (
        <View style={styles.centeredFill}>
          <ActivityIndicator color="#94a3b8" />
        </View>
      ) : scores.length === 0 ? (
        <View style={styles.centeredFill}>
          <Ionicons name="albums-outline" size={48} color="#334155" />
          <ThemedText style={styles.emptyTitle}>No recordings yet</ThemedText>
          <ThemedText style={styles.emptyText}>
            Capture a session on the Live tab, then tap Save Recording to keep it
            here.
          </ThemedText>
        </View>
      ) : (
        <FlatList
          data={scores}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.scoreRow}
              onPress={() => openScore(item.id)}
              activeOpacity={0.7}
              disabled={isLoadingDetail}
            >
              <View style={styles.scoreRowIcon}>
                <Ionicons name="musical-note" size={20} color="#7dd3fc" />
              </View>
              <View style={styles.scoreRowBody}>
                <ThemedText
                  style={styles.scoreRowTitle}
                  lightColor="#f8fafc"
                  darkColor="#f8fafc"
                  numberOfLines={1}
                >
                  {item.title}
                </ThemedText>
                <ThemedText style={styles.scoreRowMeta} numberOfLines={1}>
                  {formatTimestamp(item.createdAt)} ·{" "}
                  {item.noteCount + item.chordCount} events ·{" "}
                  {formatLength(item.durationSeconds)}
                </ThemedText>
              </View>
              <TouchableOpacity
                style={styles.scoreRowDelete}
                onPress={() => confirmDelete(item)}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Ionicons name="trash-outline" size={18} color="#64748b" />
              </TouchableOpacity>
            </TouchableOpacity>
          )}
        />
      )}

      {isLoadingDetail ? (
        <View style={styles.detailLoadingOverlay} pointerEvents="none">
          <ActivityIndicator color="#f8fafc" />
        </View>
      ) : null}
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    paddingTop: 64,
    paddingHorizontal: 16,
  },
  header: {
    gap: 6,
    marginBottom: 16,
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: "800",
    letterSpacing: -0.5,
    color: "#f8fafc",
  },
  headerSubtitle: {
    fontSize: 13,
    lineHeight: 19,
    color: "rgba(226,232,240,0.7)",
  },
  centeredFill: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    paddingHorizontal: 24,
  },
  emptyTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: "#e2e8f0",
  },
  emptyText: {
    fontSize: 13,
    lineHeight: 20,
    textAlign: "center",
    color: "rgba(148,163,184,0.85)",
  },
  listContent: {
    gap: 10,
    paddingBottom: 28,
  },
  scoreRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.18)",
    backgroundColor: "rgba(15,23,42,0.72)",
  },
  scoreRowIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(125,211,252,0.12)",
  },
  scoreRowBody: {
    flex: 1,
    gap: 4,
  },
  scoreRowTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: "#f8fafc",
  },
  scoreRowMeta: {
    fontSize: 12,
    color: "rgba(148,163,184,0.9)",
  },
  scoreRowDelete: {
    padding: 6,
  },
  detailLoadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(4,7,15,0.35)",
  },
  detailTopBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 12,
  },
  backButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 2,
  },
  backButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#f8fafc",
  },
  detailTitleWrap: {
    flex: 1,
    gap: 2,
  },
  detailTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#f8fafc",
    letterSpacing: -0.2,
  },
  detailSubtitle: {
    fontSize: 12,
    color: "rgba(148,163,184,0.9)",
  },
  detailScore: {
    flex: 1,
    borderRadius: 20,
    overflow: "hidden",
    marginBottom: 14,
    backgroundColor: "rgba(255,255,255,0.02)",
  },
  exportCard: {
    borderRadius: 24,
    padding: 16,
    gap: 14,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: "rgba(148,163,184,0.24)",
  },
  exportCardTitle: {
    fontSize: 16,
    fontWeight: "800",
    color: "#f8fafc",
    letterSpacing: -0.2,
  },
  exportRow: {
    flexDirection: "row",
    gap: 12,
  },
  exportActionButton: {
    minHeight: 48,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.18)",
    backgroundColor: "rgba(255,255,255,0.12)",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
  },
  exportActionButtonFlex: {
    flex: 1,
  },
  exportActionButtonText: {
    color: "#ffffff",
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  deleteButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 6,
  },
  deleteButtonText: {
    color: "#fca5a5",
    fontSize: 13,
    fontWeight: "700",
  },
});
