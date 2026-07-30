interface CardProps {
  children: any;
  className?: string;
  style?: any;
}
 
export default function Card({ children, className = "", style }: CardProps) {
  return (
    <div
      className={`rounded-2xl p-6 shadow-sm flex flex-col ${className}`}
      style={{
        backgroundColor: "var(--card)",
        color: "var(--card-foreground)",
        border: "1px solid #c7d4f0",
        boxShadow: "0 0 12px rgba(199,212,240,0.08)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}