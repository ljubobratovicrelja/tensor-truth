export function StreamingIndicator() {
  return (
    <div className="flex items-center gap-1 px-4 py-2">
      <div className="flex gap-1">
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full [animation-delay:-0.3s]" />
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full [animation-delay:-0.15s]" />
        <span className="bg-muted-foreground h-2 w-2 animate-bounce rounded-full" />
      </div>
      <span className="text-muted-foreground ml-2 text-sm">Thinking...</span>
    </div>
  );
}
