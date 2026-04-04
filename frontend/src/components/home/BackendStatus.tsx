import { useEffect, useState } from "react";
import { Activity, CheckCircle2, LoaderCircle, XCircle } from "lucide-react";
import Card from "../ui/Card";
import StatusBadge from "../ui/StatusBadge";
import { checkHealth } from "../../lib/api";
import type { HealthResponse } from "../../types/api";

type Status = "checking" | "online" | "offline";

export default function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [lastChecked, setLastChecked] = useState<string>("");
  const [details, setDetails] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    const runCheck = async () => {
      setLastChecked(new Date().toLocaleTimeString());
      setStatus("checking");

      try {
        const response = await checkHealth(AbortSignal.timeout(3000));
        if (cancelled) {
          return;
        }
        setDetails(response);
        setStatus(response.status === "ok" ? "online" : "offline");
      } catch {
        if (!cancelled) {
          setStatus("offline");
          setDetails(null);
        }
      }
    };

    void runCheck();
    const intervalId = window.setInterval(() => {
      void runCheck();
    }, 30000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const StatusIcon =
    status === "online" ? CheckCircle2 : status === "checking" ? LoaderCircle : XCircle;
  const statusColor =
    status === "online"
      ? "var(--accent)"
      : status === "checking"
        ? "var(--primary)"
        : "var(--destructive)";

  return (
    <Card className="!flex-row items-start gap-4">
      <div className="flex-shrink-0 mt-0.5">
        <StatusIcon
          className={`w-5 h-5${status === "checking" ? " animate-spin" : ""}`}
          style={{ color: statusColor }}
        />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Docker API status
          </span>
          <StatusBadge status={status} />
        </div>
        {lastChecked ? (
          <p className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)" }}>
            Last checked: {lastChecked}
          </p>
        ) : null}
        {details ? (
          <p className="text-xs mt-1" style={{ color: "var(--muted-foreground)" }}>
            Device: {details.device} | Pipeline ready: {details.pipelineReady ? "yes" : "no"}
          </p>
        ) : null}
      </div>
      <Activity className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "var(--muted-foreground)" }} />
    </Card>
  );
}
