export default function HeroSection() {
  return (
    <section className="px-6 pt-16 pb-2 text-center bg-transparent">
      <h2
        className="text-5xl font-semibold mb-6"
        style={{ color: "var(--foreground)" }}
      >
        Why We Built Immuna
      </h2>
      <p
        className="max-w-4xl mx-auto text-base leading-relaxed"
        style={{ color: "#bcb1da" }}
      >
        As generative AI becomes more accessible, image manipulation is no longer limited to experts. Anyone can alter photos in seconds—often without consent, context, or accountability.
        <strong> Immuna was created to shift image security from reactive detection to proactive prevention.</strong>
      </p>
    </section>
  );
}
 
