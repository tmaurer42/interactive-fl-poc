import { configureOrt } from "./ortConfig";
import { registerComponents } from "./components";

configureOrt();
registerComponents();

export * from "./components";
export * from "./modules";
