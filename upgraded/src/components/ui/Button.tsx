
interface ButtonProps {
  children: any;
  variant?: "primary" | "ghost";
  fullWidth?: boolean;
  className?: string;
  disabled?: boolean;
  onClick?: () => void;
}

export default function Button({
  children,
  variant = "primary",
  fullWidth = false,
  className = "",
  disabled,
  onClick,
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-xl px-5 py-3 text-sm transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed";

  const variants = {
    primary: { backgroundColor: "var(--secondary)", color: "var(--secondary-foreground)" },
    ghost:   { backgroundColor: "transparent",       color: "var(--accent)" },
  };

  return (
    <button
      className={`${base} ${fullWidth ? "w-full" : ""} ${className}`}
      style={variants[variant]}
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.opacity = "0.85"; }}
      onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
    >
      {children}
    </button>
  );
}
