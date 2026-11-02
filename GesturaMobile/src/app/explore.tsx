import React, { useMemo, useState } from 'react';
import {
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

interface SignEntry {
  id: string;
  name: string;
  category: 'STD' | 'ISL_ALPHA' | 'ISL_NUM';
  icon: string;
  summary: string;
  fingerGuide: string;
  tips: string;
}

const SIGN_DATABASE: SignEntry[] = [
  // Standard Model Signs
  {
    id: 'openPalm',
    name: 'Open Palm',
    category: 'STD',
    icon: '🖐️',
    summary: 'All 5 fingers extended straight and spread upward.',
    fingerGuide: 'Keep palm facing front camera, fingers straight and separated.',
    tips: 'Hold at chest height, keep fingers from overlapping.',
  },
  {
    id: 'closedFist',
    name: 'Closed Fist',
    category: 'STD',
    icon: '✊',
    summary: 'Fingers curled into a tight fist with thumb outside.',
    fingerGuide: 'Curl all 4 fingers into palm; thumb rests over index and middle.',
    tips: 'Ensure thumb is visible to the camera, not tucked inside.',
  },
  {
    id: 'fingerCircle',
    name: 'Finger Circle (OK)',
    category: 'STD',
    icon: '👌',
    summary: 'Thumb tip and index finger tip touching to form a circular loop.',
    fingerGuide: 'Form circle with thumb and index; extend middle, ring, and pinky.',
    tips: 'Ensure the circle opening is clearly facing the lens.',
  },
  {
    id: 'fingerSymbols',
    name: 'Finger Symbols (Peace / V)',
    category: 'STD',
    icon: '✌️',
    summary: 'Index and middle fingers extended apart in a V shape.',
    fingerGuide: 'Extend index and middle; tuck ring and pinky under thumb.',
    tips: 'Spread fingers in a clear V with palm facing camera.',
  },
  {
    id: 'singleFingerBend',
    name: 'Single Finger Bend',
    category: 'STD',
    icon: '☝️',
    summary: 'Index finger pointing upward, other fingers closed.',
    fingerGuide: 'Extend index finger vertically; thumb locks middle and ring.',
    tips: 'Keep index finger straight up within the reticle.',
  },
  {
    id: 'semiOpenPalm',
    name: 'Semi-Open Palm',
    category: 'STD',
    icon: '👋',
    summary: 'Hand open in a relaxed, gentle cup posture.',
    fingerGuide: 'Fingers slightly curved inward naturally, palm facing camera.',
    tips: 'Avoid curling fingers completely.',
  },
  {
    id: 'semiOpenFist',
    name: 'Semi-Open Fist',
    category: 'STD',
    icon: '🤛',
    summary: 'Fingers loosely curled without making full tension.',
    fingerGuide: 'Fingers curled halfway toward palm with thumb beside them.',
    tips: 'Keep wrist straight in line with forearm.',
  },
  {
    id: 'multiFingerBend',
    name: 'Multi-Finger Bend',
    category: 'STD',
    icon: '🤏',
    summary: 'Multiple fingers curled at middle knuckles like a claw.',
    fingerGuide: 'Bend first two knuckles of all fingers while keeping palm open.',
    tips: 'Keep hand at a slight 45-degree angle to reveal knuckle bends.',
  },

  // Indian Sign Language: Numbers (1-9)
  {
    id: 'num_1',
    name: 'Number 1',
    category: 'ISL_NUM',
    icon: '1️⃣',
    summary: 'One finger (index) extended vertically.',
    fingerGuide: 'Extend index finger upright, keep others tucked.',
    tips: 'Keep hand stationary for best detection.',
  },
  {
    id: 'num_2',
    name: 'Number 2',
    category: 'ISL_NUM',
    icon: '2️⃣',
    summary: 'Index and middle fingers extended upward.',
    fingerGuide: 'Extend index and middle fingers together or in a slight V.',
    tips: 'Thumb holds ring and pinky in place.',
  },
  {
    id: 'num_3',
    name: 'Number 3',
    category: 'ISL_NUM',
    icon: '3️⃣',
    summary: 'Thumb, index, and middle extended (or 3 middle fingers).',
    fingerGuide: 'Raise three fingers straight up, tuck pinky and ring.',
    tips: 'Keep fingers clearly separated.',
  },
  {
    id: 'num_4',
    name: 'Number 4',
    category: 'ISL_NUM',
    icon: '4️⃣',
    summary: 'Four fingers extended upward, thumb folded across palm.',
    fingerGuide: 'Extend all four fingers, tuck thumb across inner palm.',
    tips: 'Hold fingers upright facing the camera.',
  },
  {
    id: 'num_5',
    name: 'Number 5',
    category: 'ISL_NUM',
    icon: '5️⃣',
    summary: 'Full open palm with all five fingers extended.',
    fingerGuide: 'Spread all 5 fingers open, palm forward.',
    tips: 'Matches the open palm standard sign.',
  },
  {
    id: 'num_6',
    name: 'Number 6',
    category: 'ISL_NUM',
    icon: '6️⃣',
    summary: 'Thumb and pinky touching, three middle fingers raised.',
    fingerGuide: 'Touch thumb to pinky tip, extend index, middle, and ring.',
    tips: 'Ensure 3 raised fingers are clearly visible.',
  },
  {
    id: 'num_7',
    name: 'Number 7',
    category: 'ISL_NUM',
    icon: '7️⃣',
    summary: 'Thumb touching ring finger tip.',
    fingerGuide: 'Touch thumb to ring finger, extend index, middle, and pinky.',
    tips: 'Keep other 3 fingers straight.',
  },
  {
    id: 'num_8',
    name: 'Number 8',
    category: 'ISL_NUM',
    icon: '8️⃣',
    summary: 'Thumb touching middle finger tip.',
    fingerGuide: 'Touch thumb to middle finger, extend index, ring, and pinky.',
    tips: 'Ensure ring and pinky remain erect.',
  },
  {
    id: 'num_9',
    name: 'Number 9',
    category: 'ISL_NUM',
    icon: '9️⃣',
    summary: 'Thumb touching index finger tip.',
    fingerGuide: 'Touch thumb to index finger, other 3 fingers upright.',
    tips: 'Similar to the OK sign.',
  },

  // Indian Sign Language: Selected Alphabet
  {
    id: 'isl_A',
    name: 'Letter A',
    category: 'ISL_ALPHA',
    icon: '🅰️',
    summary: 'Fist with thumb resting straight up against side of index.',
    fingerGuide: 'Make a fist; keep thumb alongside index finger pointing up.',
    tips: 'Thumb must point vertically, not curled over fingers.',
  },
  {
    id: 'isl_B',
    name: 'Letter B',
    category: 'ISL_ALPHA',
    icon: '🅱️',
    summary: 'Four fingers straight up together, thumb folded across palm.',
    fingerGuide: 'Keep 4 fingers upright and joined; thumb folded on palm.',
    tips: 'Fingers should touch each other, not spread.',
  },
  {
    id: 'isl_C',
    name: 'Letter C',
    category: 'ISL_ALPHA',
    icon: '©️',
    summary: 'Hand curved into a C shape facing sideways.',
    fingerGuide: 'Curve all fingers and thumb to mimic letter C silhouette.',
    tips: 'Angle hand slightly sideways so camera sees the curve.',
  },
  {
    id: 'isl_D',
    name: 'Letter D',
    category: 'ISL_ALPHA',
    icon: '🇩',
    summary: 'Index finger pointing straight up, other fingers forming circle with thumb.',
    fingerGuide: 'Index points up; thumb touches middle, ring, and pinky tips.',
    tips: 'Make sure only index finger is extended.',
  },
  {
    id: 'isl_L',
    name: 'Letter L',
    category: 'ISL_ALPHA',
    icon: '🇱',
    summary: 'Thumb and index finger form a 90-degree L shape.',
    fingerGuide: 'Extend thumb out and index straight up; others curled.',
    tips: 'Form a sharp right angle between index and thumb.',
  },
  {
    id: 'isl_V',
    name: 'Letter V',
    category: 'ISL_ALPHA',
    icon: '✌️',
    summary: 'Index and middle fingers in a V shape.',
    fingerGuide: 'Raise index and middle fingers apart; thumb holds others.',
    tips: 'Keep fingers straight and well-separated.',
  },
  {
    id: 'isl_Y',
    name: 'Letter Y',
    category: 'ISL_ALPHA',
    icon: '🤙',
    summary: 'Thumb and pinky extended out, three middle fingers curled.',
    fingerGuide: 'Extend thumb and pinky sideways; curl middle 3 fingers.',
    tips: 'Classic "hang loose" or phone silhouette.',
  },
];

type CategoryFilter = 'ALL' | 'STD' | 'ISL_ALPHA' | 'ISL_NUM';

export default function ExploreScreen() {
  const insets = useSafeAreaInsets();
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState<CategoryFilter>('ALL');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const filteredSigns = useMemo(() => {
    return SIGN_DATABASE.filter((item) => {
      const matchCategory = category === 'ALL' || item.category === category;
      const matchSearch =
        item.name.toLowerCase().includes(search.toLowerCase()) ||
        item.summary.toLowerCase().includes(search.toLowerCase());
      return matchCategory && matchSearch;
    });
  }, [search, category]);

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Sign Dictionary</Text>
        <Text style={styles.subtitle}>
          Learn and practice the gestures recognized by Gestura AI
        </Text>

        {/* Search Bar */}
        <View style={styles.searchContainer}>
          <Text style={styles.searchIcon}>🔍</Text>
          <TextInput
            style={styles.searchInput}
            placeholder="Search gestures, letters, or numbers..."
            placeholderTextColor="#888"
            value={search}
            onChangeText={setSearch}
          />
          {search.length > 0 && (
            <Pressable onPress={() => setSearch('')}>
              <Text style={styles.clearSearch}>✕</Text>
            </Pressable>
          )}
        </View>

        {/* Category Pills */}
        <View style={styles.pillsRow}>
          {(
            [
              { id: 'ALL', label: 'All Signs' },
              { id: 'STD', label: 'Standard (8)' },
              { id: 'ISL_ALPHA', label: 'ISL Alphabet' },
              { id: 'ISL_NUM', label: 'Numbers (1-9)' },
            ] as const
          ).map((pill) => (
            <Pressable
              key={pill.id}
              style={[styles.pill, category === pill.id && styles.pillActive]}
              onPress={() => setCategory(pill.id)}
            >
              <Text style={[styles.pillText, category === pill.id && styles.pillTextActive]}>
                {pill.label}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      {/* Signs List */}
      <FlatList
        data={filteredSigns}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => {
          const isExpanded = expandedId === item.id;
          return (
            <Pressable
              style={[styles.card, isExpanded && styles.cardExpanded]}
              onPress={() => setExpandedId(isExpanded ? null : item.id)}
            >
              <View style={styles.cardHeader}>
                <Text style={styles.cardIcon}>{item.icon}</Text>
                <View style={styles.cardInfo}>
                  <View style={styles.cardTitleRow}>
                    <Text style={styles.cardTitle}>{item.name}</Text>
                    <View style={styles.categoryBadge}>
                      <Text style={styles.categoryBadgeText}>{item.category}</Text>
                    </View>
                  </View>
                  <Text style={styles.cardSummary}>{item.summary}</Text>
                </View>
              </View>

              {isExpanded && (
                <View style={styles.cardDetails}>
                  <View style={styles.detailSection}>
                    <Text style={styles.detailLabel}>✋ Finger Placement:</Text>
                    <Text style={styles.detailText}>{item.fingerGuide}</Text>
                  </View>
                  <View style={styles.detailSection}>
                    <Text style={styles.detailLabel}>💡 Camera Pro-Tip:</Text>
                    <Text style={styles.detailText}>{item.tips}</Text>
                  </View>
                </View>
              )}
            </Pressable>
          );
        }}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyEmoji}>🔎</Text>
            <Text style={styles.emptyTitle}>No signs found</Text>
            <Text style={styles.emptySubtitle}>Try searching for a different keyword or category</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F1016',
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: -0.5,
  },
  subtitle: {
    fontSize: 14,
    color: '#8A8F9E',
    marginTop: 4,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1C1E29',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#2A2D3D',
  },
  searchIcon: {
    fontSize: 16,
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    color: '#FFFFFF',
    fontSize: 15,
  },
  clearSearch: {
    color: '#8A8F9E',
    fontSize: 14,
    paddingHorizontal: 6,
  },
  pillsRow: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 14,
    flexWrap: 'wrap',
  },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 20,
    backgroundColor: '#1C1E29',
    borderWidth: 1,
    borderColor: '#2A2D3D',
  },
  pillActive: {
    backgroundColor: '#0055FF',
    borderColor: '#3377FF',
  },
  pillText: {
    color: '#8A8F9E',
    fontSize: 12,
    fontWeight: '600',
  },
  pillTextActive: {
    color: '#FFFFFF',
  },
  listContent: {
    paddingHorizontal: 20,
    paddingBottom: 40,
    gap: 12,
  },
  card: {
    backgroundColor: '#181A24',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#252837',
  },
  cardExpanded: {
    borderColor: '#0055FF88',
    backgroundColor: '#1B1E2B',
  },
  cardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
  },
  cardIcon: {
    fontSize: 32,
  },
  cardInfo: {
    flex: 1,
  },
  cardTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  cardTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  categoryBadge: {
    backgroundColor: '#0055FF22',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#0055FF55',
  },
  categoryBadgeText: {
    color: '#6699FF',
    fontSize: 10,
    fontWeight: '700',
  },
  cardSummary: {
    fontSize: 13,
    color: '#9E9EB2',
    lineHeight: 18,
  },
  cardDetails: {
    marginTop: 14,
    paddingTop: 14,
    borderTopWidth: 1,
    borderTopColor: '#282C3D',
    gap: 10,
  },
  detailSection: {
    gap: 2,
  },
  detailLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: '#00FFCC',
  },
  detailText: {
    fontSize: 13,
    color: '#D1D5DB',
    lineHeight: 18,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 60,
  },
  emptyEmoji: {
    fontSize: 42,
  },
  emptyTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
    marginTop: 12,
  },
  emptySubtitle: {
    color: '#8A8F9E',
    fontSize: 13,
    marginTop: 4,
  },
});
