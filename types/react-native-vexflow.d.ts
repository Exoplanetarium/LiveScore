declare module 'react-native-vexflow' {
  export function useScore(config: {
    contextSize: { x: number; y: number };
    staveOffset: { x: number; y: number };
    staveWidth: number;
    clef: string;
    timeSig: string;
  }): [any, any];
}
