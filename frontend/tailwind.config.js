/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        rbu: {
          50: "#eef6ff",
          100: "#d8e8ff",
          200: "#afceff",
          300: "#7aa9ff",
          400: "#467ff8",
          500: "#265fe4",
          600: "#1a4bc2",
          700: "#143b98",
          800: "#102f74",
          900: "#0c2457"
        }
      },
      boxShadow: {
        panel: "0 20px 40px -24px rgba(12, 36, 87, 0.55)"
      }
    }
  },
  plugins: []
};
