export function Logo({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="fl-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#c7cfff" />
          <stop offset="1" stopColor="#5b6cff" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#fl-g)" />
      {[7, 11, 15, 19, 23].map((x, i) => (
        <rect key={x} x={x} y={16 - [3, 7, 10, 6, 3][i]} width="2.2" height={[3, 7, 10, 6, 3][i] * 2} rx="1.1"
              fill="#0b0b14" />
      ))}
    </svg>
  );
}
