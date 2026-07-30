import Card from "../ui/Card";

const META = [
  { label: "Institution", value: "Southern Alberta Institute of Technology (SAIT)" },
  { label: "Program",     value: "Information Systems Security" },
  { label: "Year",        value: "2026" },
];

export default function AcademicContext() {
  return (
    <Card>
      <h2 className="text-lg font-semibold mb-5" style={{ color: "var(--foreground)" }}>
        Academic Context
      </h2>

      <div className="space-y-4 text-base leading-relaxed" style={{ color: "#bcb1da" }}>
        <p>
          Immuna is a cybersecurity capstone project developed at the
          Southern Alberta Institute of Technology (SAIT).
        </p>
        <p>
          This work explores how security principles—traditionally applied to systems and networks—can 
          be extended to media itself, particularly in the context of generative AI.
        </p>
        <p>
          Our focus is on:

          <ul className="list-disc ml-6 space-y-1">
          <li>Proactive defense against AI misuse.</li>
          <li>Privacy-preserving system design.</li>
          <li>Secure, containerized processing architectures.</li>
        </ul>

        <br />
          As synthetic media continues to evolve, protecting digital 
          content at the source becomes increasingly critical.
        </p>
      </div>

      <div
        className="mt-2 pt-5 border-t space-y-1"
        style={{ borderColor: "var(--border)" }}
      >
        {META.map(({ label, value }) => (
          <p key={label} className="text-base" style={{ color: "#bcb1da" }}>
            <span className="font-semibold" style={{ color: "var(--foreground)" }}>
              {label}:
            </span>{" "}
            {value}
          </p>
        ))}
      </div>
    </Card>
  );
}
