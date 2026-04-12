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
        border: "1px solid var(--border)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}