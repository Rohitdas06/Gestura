import React from 'react';
import { StyleSheet, View, Text } from 'react-native';

export interface Point2D {
  x: number;
  y: number;
}

// Canonical MediaPipe hand skeleton connections (21 joints, 21 bones)
export const SKELETON_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm base closure
  [0, 17],
];

// Tip landmark indices
const FINGERTIPS = [4, 8, 12, 16, 20];

interface Props {
  landmarks: number[] | null;
  width: number;
  height: number;
  mirrorX: boolean;
  handDetected: boolean;
}

export const HandSkeletonOverlay = React.memo(function HandSkeletonOverlay({
  landmarks,
  width,
  height,
  mirrorX,
  handDetected,
}: Props) {
  // Center Framing Reticle dimensions
  const boxWidth = Math.min(width * 0.72, 280);
  const boxHeight = Math.min(height * 0.5, 340);
  const boxLeft = (width - boxWidth) / 2;
  const boxTop = (height - boxHeight) / 2 - 30;

  // Convert normalized 63 landmarks (21 x 3) into screen points
  const points: Point2D[] = [];
  if (landmarks && landmarks.length >= 63 && width > 0 && height > 0) {
    for (let i = 0; i < 21; i++) {
      const normX = landmarks[i * 3];
      const normY = landmarks[i * 3 + 1];
      const px = (mirrorX ? 1 - normX : normX) * width;
      const py = normY * height;
      points.push({ x: px, y: py });
    }
  }

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* 1. Camera Framing Reticle / Target Box */}
      <View
        style={[
          styles.reticle,
          {
            left: boxLeft,
            top: boxTop,
            width: boxWidth,
            height: boxHeight,
            borderColor: handDetected ? '#00FF66' : 'rgba(255, 255, 255, 0.25)',
          },
        ]}
      >
        {/* Reticle corner accents */}
        <View style={[styles.corner, styles.cornerTL, handDetected && styles.cornerActive]} />
        <View style={[styles.corner, styles.cornerTR, handDetected && styles.cornerActive]} />
        <View style={[styles.corner, styles.cornerBL, handDetected && styles.cornerActive]} />
        <View style={[styles.corner, styles.cornerBR, handDetected && styles.cornerActive]} />

        {/* Center Target Label */}
        <View style={styles.reticleBadge}>
          <Text style={[styles.reticleText, handDetected && styles.reticleTextActive]}>
            {handDetected ? 'HAND LOCKED' : 'PLACE HAND HERE'}
          </Text>
        </View>
      </View>

      {/* 2. Bones (Connecting Lines) */}
      {points.length === 21 &&
        SKELETON_CONNECTIONS.map(([startIdx, endIdx], idx) => {
          const p1 = points[startIdx];
          const p2 = points[endIdx];
          if (!p1 || !p2) return null;

          const dx = p2.x - p1.x;
          const dy = p2.y - p1.y;
          const length = Math.sqrt(dx * dx + dy * dy);
          const angle = Math.atan2(dy, dx) * (180 / Math.PI);
          const midX = (p1.x + p2.x) / 2;
          const midY = (p1.y + p2.y) / 2;

          return (
            <View
              key={`bone-${idx}`}
              style={{
                position: 'absolute',
                left: midX - length / 2,
                top: midY - 1.5,
                width: length,
                height: 3,
                backgroundColor: 'rgba(0, 255, 200, 0.75)',
                borderRadius: 1.5,
                transform: [{ rotate: `${angle}deg` }],
              }}
            />
          );
        })}

      {/* 3. Joints (Landmark Dots) */}
      {points.map((pt, idx) => {
        const isTip = FINGERTIPS.includes(idx);
        const isWrist = idx === 0;
        const size = isTip ? 12 : isWrist ? 14 : 8;
        const color = isTip ? '#FF0055' : isWrist ? '#FFCC00' : '#00FFFF';

        return (
          <View
            key={`joint-${idx}`}
            style={{
              position: 'absolute',
              left: pt.x - size / 2,
              top: pt.y - size / 2,
              width: size,
              height: size,
              borderRadius: size / 2,
              backgroundColor: color,
              borderWidth: 1.5,
              borderColor: '#FFFFFF',
              shadowColor: color,
              shadowRadius: 4,
              shadowOpacity: 0.8,
            }}
          />
        );
      })}
    </View>
  );
});

const styles = StyleSheet.create({
  reticle: {
    position: 'absolute',
    borderWidth: 1.5,
    borderRadius: 24,
    borderStyle: 'dashed',
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 12,
  },
  reticleBadge: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  reticleText: {
    color: 'rgba(255, 255, 255, 0.65)',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.2,
  },
  reticleTextActive: {
    color: '#00FF66',
  },
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: 'rgba(255, 255, 255, 0.7)',
  },
  cornerActive: {
    borderColor: '#00FF66',
  },
  cornerTL: {
    top: -2,
    left: -2,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: 18,
  },
  cornerTR: {
    top: -2,
    right: -2,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: 18,
  },
  cornerBL: {
    bottom: -2,
    left: -2,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: 18,
  },
  cornerBR: {
    bottom: -2,
    right: -2,
    borderBottomWidth: 4,
    borderRightWidth: 4,
    borderBottomRightRadius: 18,
  },
});
