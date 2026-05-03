import { Link, useLocation } from "wouter";
import { ShieldCheck, LayoutDashboard, Search, Database } from "lucide-react";

export function Navigation() {
  const [location] = useLocation();

  const links = [
    { href: "/", label: "Analyzer", icon: Search },
    { href: "/demo", label: "Demo Lab", icon: Database },
    { href: "/dashboard", label: "Intel Dashboard", icon: LayoutDashboard },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-slate-950/80 backdrop-blur-md">
      <div className="container mx-auto flex h-16 items-center px-4">
        <div className="flex items-center gap-2 font-bold text-lg tracking-tight text-white mr-8">
          <ShieldCheck className="h-6 w-6 text-blue-500" />
          <span>Verity<span className="text-blue-500 text-sm font-normal ml-1">AI</span></span>
        </div>
        <nav className="flex items-center gap-6">
          {links.map((link) => {
            const Icon = link.icon;
            const isActive = location === link.href;
            return (
              <Link 
                key={link.href} 
                href={link.href}
                className={`flex items-center gap-2 text-sm font-medium transition-colors hover:text-white ${isActive ? "text-white" : "text-white/60"}`}
              >
                <Icon className="h-4 w-4" />
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
