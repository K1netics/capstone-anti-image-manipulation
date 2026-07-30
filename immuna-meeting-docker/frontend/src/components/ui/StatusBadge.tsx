type Status = "checking" | "online" | "offline";

interface StatusBadgeProps {
  status: Status;
}

const config: Record<Status, { dotColor: string; label: string }> = {
  checking: { dotColor: "#f59e0b", label: "Checking…" },
  online:   { dotColor: "#22c55e", label: "API Online" },
  offline:  { dotColor: "#d4183d", label: "Unable to reach API" },
};

export default function StatusBadge({ status }: StatusBadgeProps) {
  const { dotColor, label } = config[status];

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="w-2 h-2 rounded-full animate-pulse flex-shrink-0"
        style={{ backgroundColor: dotColor }}
      />
      <span className="text-xs" style={{ color: "var(--muted-foreground)" }}>
        {label}
      </span>
    </span>
  );
}
