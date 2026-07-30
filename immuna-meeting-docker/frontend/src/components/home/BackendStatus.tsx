import { useCallback, useEffect, useState } from "react";
import { Activity, XCircle } from "lucide-react";
import Card from "../ui/Card";
import StatusBadge from "../ui/StatusBadge";
 
type Status = "checking" | "online" | "offline";
 
export default function BackendStatus() {
  const [status, setStatus] = useState<Status>("checking");
  const [lastChecked, setLastChecked] = useState<string>("");
 
  const check = useCallback(async () => {
    setLastChecked(new Date().toLocaleTimeString());
    try {
      const res = await fetch("http://localhost:8000/health", {
        signal: AbortSignal.timeout(3000),
      });
      setStatus(res.ok ? "online" : "offline");
    } catch {
      setStatus("offline");
    }
  }, []);
 
  useEffect(() => {
    check();
    const id = setInterval(check, 30_000);
    return () => clearInterval(id);
  }, [check]);
 
  return (
    <Card className="!flex-row items-start gap-4">
      <div className="flex-shrink-0 mt-0.5">
        <XCircle className="w-5 h-5" style={{ color: "var(--destructive)" }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Backend Status
          </span>
          <StatusBadge status={status} />
        </div>
        {lastChecked && (
          <p className="text-xs mt-0.5" style={{ color: "var(--muted-foreground)" }}>
            Last checked: {lastChecked}
          </p>
        )}
      </div>
      <Activity className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: "var(--muted-foreground)" }} />
    </Card>
  );
}
