type Status = "checking" | "online" | "offline";

interface StatusBadgeProps {
  status: Status;
}

const config: Record<Status, { dotColor: string; label: string }> = {
  checking: { dotColor: "var(--primary)", label: "Checking API" },
  online: { dotColor: "var(--accent)", label: "API online" },
  offline: { dotColor: "var(--destructive)", label: "API offline" },
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
